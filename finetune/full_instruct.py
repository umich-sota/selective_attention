# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from lightning.fabric.strategies import FSDPStrategy,DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.model import GPT, Block, Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt

# Hyperparameters
# iterations
max_iters = 5000
warmup_iters = 1000
eval_interval = 100
save_interval = 1000
eval_iters = 100
eval_max_new_tokens = 100
log_interval = 1
FSDP=False
COSINE_LR = False
WANDB = True

# Trainer related
batch_size = 64 
micro_batch_size = 16
gradient_accumulation_steps = 0 # will compute it later in setup() = bach_size / micro_batch_size / devices
max_seq_length = None  # assign value to truncate

# opimizer
learning_rate = 1e-5
min_lr = 1e-6
weight_decay = 0.01
lr_decay_iters = max_iters
beta1 = 0.9
beta2 = 0.95
decay_lr = False

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/full/alpaca"),
    name: str = "stablelm-base-alpha-3b",
    config_file = "config.json",
    precision: Optional[str] = None,
    resume: Union[bool, Path] = False,
    devices: int = 1,
) -> None:
    precision = precision or get_default_supported_precision(training=True)

    fabric_devices = devices
    if fabric_devices > 1:
        if FSDP:
        # For large models, FSDP is recommended to avoid OOMs
            print("Using FSDP")
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
        # For samll model in debug, we can use DDP
        else:
            print("Using DDP")
            parallel_devices = [torch.device(f"cuda:{i}") for i in range(devices)]
            strategy = DDPStrategy( parallel_devices=parallel_devices, precision=precision)
    else:
        strategy = "auto"

    # log to local out directory
    if not WANDB:
        logger = CSVLogger("out", name, flush_logs_every_n_steps=log_interval)
    else:
        logger = WandbLogger(name, save_dir="out", project="paramAttn" )

    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)

    if hparams["gradient_accumulation_steps"] ==0:
        assert batch_size % (micro_batch_size*devices) == 0
        gradient_accumulation_steps = batch_size // (micro_batch_size * devices)
        assert gradient_accumulation_steps > 0
        hparams["gradient_accumulation_steps"] = gradient_accumulation_steps
    hparams["devices"] = devices
    hparams["name"] = name
    
    fabric.print(hparams)
    fabric.launch(main,
                  data_dir,
                  checkpoint_dir,
                  out_dir,
                  config_file=config_file,
                  name=name,
                  resume=resume,
                  hparams=hparams)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path,
         config_file: str, name: str, resume: Union[bool, Path],
         hparams: dict) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)
    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")
    config = Config.from_json(Path(config_file))
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    model.apply(model._init_weights)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")

    # model = fabric.setup_module(model)
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)
    state = {
        "model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0, "total_lengths": 0
    }

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, state, train_data, val_data, checkpoint_dir, out_dir)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_finetuned.pth"
    save_checkpoint(fabric, {"model": state["model"]}, save_path)


def train(
    fabric: L.Fabric,
    state: Dict,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, val_data, tokenizer, max_iters=2)  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    total_t0 = time.perf_counter()
    total_lengths = 0

    for state["iter_num"] in range(state["iter_num"], max_iters):
        iter_num = state["iter_num"]
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, max_iters=eval_iters)
            train_loss = validate(fabric, model, train_data, tokenizer, max_iters=eval_iters)
            t1 = time.perf_counter() - t0
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, train loss {train_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict(metrics = {"eval/val_loss": val_loss.item(), 
                                       "eval/time": t1 * 1000,
                                       "eval/train_loss": train_loss.item(),
                                       "eval/lr": lr,
                                       "step": iter_num,
                                       "eval/valtime": t1 * 1000}, step=iter_num//log_interval)
            fabric.barrier()
        if iter_num % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_checkpoint(fabric, state, checkpoint_path)        

        # Accumulate gradients over multiple micro-batches
        iter_t0 = time.perf_counter()
        iter_length = 0
        gradient_accumulation_steps = hparams["gradient_accumulation_steps"]
        for micro_step in range(gradient_accumulation_steps):
            
            is_accumulating = micro_step == gradient_accumulation_steps - 1
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                input_ids, targets = get_batch(fabric, train_data, longest_seq_ix if iter_num == 1 else None)
                logits = model(input_ids)
                
                # shift the targets such that output n predicts token n+1
                loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
                fabric.backward(loss / gradient_accumulation_steps)
                iter_length += input_ids.numel()
                # print(is_accumulating, logits.shape, targets.shape, input_ids.shape, targets.shape, input_ids.numel(),iter_length)
        optimizer.step()
        optimizer.zero_grad()
        total_lengths += iter_length
        if iter_num % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * batch_size,
                lengths=total_lengths
            )
            t_used = t1 - total_t0
            est_time = t_used / (iter_num + 1) * max_iters - t_used
            fabric.log_dict(metrics = {"running/iter": iter_num,
                                       "running/loss": loss_item,
                                       "running/lr": lr, 
                                        "running/remaining_time": est_time / 60. / 60.,
                                       "running/itertime": (t1 - iter_t0) * 1000,
                                       "step": iter_num}, step=iter_num//log_interval)
            fabric.print(
                f"iter {iter_num}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms, est. time remaining: {est_time / 60. / 60.:.2f}h"
            )

# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(max_iters)
    for k in range(max_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    # fabric.print(instruction)
    # sample = {"instruction": instruction, "input": ""}
    # prompt = generate_prompt(sample)
    # encoded = tokenizer.encode(prompt, device=fabric.device)
    # with fabric.init_tensor():
    #     # do not set `max_seq_length=max_returned_token` because memory is not a concern here
    #     model.set_kv_cache(batch_size=1)
    # output = generate(
    #     model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id
    # )
    # model.clear_kv_cache()
    # output = tokenizer.decode(output)
    # fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # Truncate if needed
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_checkpoint(fabric, state, file_path: Path):
    fabric.print(f"Saving weights to {str(file_path)!r}")
    fabric.save(file_path, state)

# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    if not COSINE_LR:
        return learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
