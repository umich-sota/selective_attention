# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightning.fabric.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from lightning.fabric.strategies import FSDPStrategy,DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from torch.utils.data import DataLoader, IterableDataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.model_statistic import GPT, Block, Config
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
# max_iters = 10000
max_iters = 100
max_iters = 5
warmup_iters = 200
eval_interval = 100
save_interval = 1000
eval_iters = 1
eval_max_new_tokens = 100
log_interval = 1
FSDP=False
COSINE_LR = False
WANDB = False

# Trainer related
batch_size = 64
micro_batch_size = 8
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

    train_data, val_data = load_datasets(data_dir, max_seq_length=model.max_seq_length)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=micro_batch_size, num_workers=2)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, checkpoint_dir, out_dir)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # # Save the final checkpoint at the end of training
    # save_path = out_dir / "lit_model_finetuned.pth"
    # save_checkpoint(fabric, {"model": state["model"]}, save_path)


def train(
    fabric: L.Fabric,
    state: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    # alphas = torch.logspace(-7, 0, 20)
    # # betas = torch.linspace(0., 3., 10)
    # # alphas = torch.linspace(0.85, 0.95, 6)
    # # betas = torch.linspace(0., 0.1, 10)
    # betas = [1]
    # result = np.zeros((len(alphas), len(betas)))
    # best_loss = float("inf")
    # best_alpha_beta = (0, 0)
    # for i, alpha in enumerate(alphas):
    #     for j, beta in enumerate(betas):
    #         model.set_value_to_block_buffers_by_name("alpha", alpha)
    #         model.set_value_to_block_buffers_by_name("beta", beta)
    #         # if result[i-1, j]>4.5:
    #         #     result[i, j] = result[i-1, j]
    #         #     continue
    #         # if result [i, j-1] > 5.5:
    #         #     result[i, j] = result[i, j-1]
    #         #     continue
    #         fabric.seed_everything(42 + fabric.global_rank)
    #         loss = validate(fabric, model, train_dataloader, max_iters=max_iters)
    #         fabric.barrier()
    #         fabric.all_reduce(loss, reduce_op="mean")
    #         fabric.print(f"alpha: {alpha}, beta: {beta}, train loss: {loss.item():.4f}, perplexity: {math.exp(loss.item()):.4f}")
    #         result[i, j] = loss.item()
    #         if loss.item() < best_loss:
    #             best_loss = loss.item()
    #             best_alpha_beta = (alpha, beta)
    # np.save("out/fig/alpha_beta.npy", result)

    # # best_alpha_beta = (1.0, -0.333)
    # best_alpha_beta = (1.0, 0.0)
    # #best_alpha_beta = (0.0015, 0.555)
    # #best_alpha_beta = (0.949, 0.022)
    # best_alpha_beta = (0.88, 0.033)
    # model.set_value_to_block_buffers_by_name("alpha", best_alpha_beta[0])
    # model.set_value_to_block_buffers_by_name("beta", best_alpha_beta[1])


    model.reset_block_buffers_by_name("sorted_att_buffer")
    loss = validate(fabric, model, train_dataloader, max_iters=max_iters)  # sanity check
    # print perplexity
    fabric.print(f"Initial train loss: {loss.item():.4f}, perplexity: {math.exp(loss.item()):.4f}")

    sorted_attention = model.get_block_buffers_by_name("sorted_att_buffer")/max_iters
    # merge all top_k_prob from all devices
    fabric.barrier()

    # fabric.all_reduce(sorted_attention, reduce_op="mean")
    if fabric.global_rank == 0:
        
        # top_k_prob = top_k_prob.cpu().numpy()
        # 
        # # top_k_prob.shape=(layers, dim), we want to plot layers lines, each line polt dim points
        # # use cool color map, draw top_k_prob for each layer
        # # high resolution
        # plt.figure(figsize=(20, 10))
        # def smooth_moving_average(x, w):
        #     # start from 100 to avoid the first 100 points
        #     need_smooth = x[50:]
        #     smoothed = np.convolve(need_smooth, np.ones(w), 'valid') / w
        #     return np.concatenate([x[:50], smoothed])
        # for i in range(top_k_prob.shape[0]):
        #     # color = plt.cm.tab20c(i / top_k_prob.shape[0])
        #     color = plt.cm.tab20b(i)
        #     # smoothed = smooth_moving_average(top_k_prob[i], 5)
        #     # plt.plot(smoothed, color=color, label=f"layer {i}", linewidth=1)
        #     plt.plot(top_k_prob[i], color=color, label=f"layer {i}", linewidth=1)
        # avg_top_k_prob = top_k_prob.mean(axis=0)
        # plt.plot(avg_top_k_prob, color="tab:red", label="average", linewidth=1)
        # # make the legend outside the plot
        # plt.legend(fontsize=17, loc="lower left", bbox_to_anchor=(1, 0))
        # plt.xscale("symlog")
        # # make x show more grid rather than only 10^0, 10^1, 10^2, 10^3
        # # Major ticks at each decade
        # major_ticks = [10**i for i in range(4)]  # 10^0, 10^1, 10^2, 10^3

        # # Additional desired ticks
        # additional_ticks = [0, 1,2,3,4,5,6,7,8,9,
        #                     20,30,40,50,60,70,80,90,
        #                     200,300,400,500,600,700,800,900,
        #                     2000]  # Manual additional ticks
        # all_ticks = sorted(major_ticks + additional_ticks)
        # plt.xticks(all_ticks)


        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        
        # plt.xlim(0, top_k_prob.shape[1])
    
        # plt.grid(linestyle="--", linewidth=0.5)
        # for tick in additional_ticks:
        #     plt.axvline(x=tick, linestyle='--', color='grey', linewidth=0.5, alpha=0.5)  # Draw lines manually for minor ticks


        # os.makedirs("out/fig/", exist_ok=True)
        # title = "top_5p_prob"
        # plt.title(title, fontsize=20)
        # plt.savefig(f"out/fig/{title}.png")
        # plt.xscale("linear")
        # plt.savefig(f"out/fig/{title}_linear.png")
        # np.save(f"out/fig/{title}.npy", top_k_prob)

        # plt.close()

        # # plot last_row_prob
        # last_row_prob = last_row_prob.cpu().numpy()
        # np.save("out/fig/last_row_prob.npy", last_row_prob)
        # plt.figure(figsize=(20, 10))
        # for i in range(last_row_prob.shape[0]):
        #     color = plt.cm.tab20b(i)
        #     plt.plot(last_row_prob[i], color=color, label=f"layer {i}", linewidth=1)
        # avg_last_row_prob = last_row_prob.mean(axis=0)
        # plt.plot(avg_last_row_prob, color="tab:red", label="average", linewidth=1)
        # plt.legend(fontsize=17, loc="lower left", bbox_to_anchor=(1, 0))
        # plt.xscale("symlog")
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlim(0, last_row_prob.shape[1])
        # plt.grid(linestyle="--", linewidth=0.5)
        # for tick in additional_ticks:
        #     plt.axvline(x=tick, linestyle='--', color='grey', linewidth=0.5, alpha=0.5)
        # title = "last_row_prob"
        # plt.title(title, fontsize=20)
        # plt.savefig(f"out/fig/{title}_x_symlog.png")

        # plt.xscale("linear")
        # plt.savefig(f"out/fig/{title}_linear.png")
        # plt.yscale("log")
        # plt.xscale("symlog")
        # # plt.ylim(0, 1.1)
        # plt.savefig(f"out/fig/{title}_xy_symlog.png")
        # plt.xscale("linear")
        # plt.savefig(f"out/fig/{title}_y_symlog.png")
        # plt.close()

        sorted_attention = sorted_attention.cpu()
        # check nan 
        nan_location = torch.argwhere(torch.isnan(sorted_attention))
        print("nan_number: ", len(nan_location))
        # if BFloat16, we need to convert it to float32
        if sorted_attention.dtype == torch.bfloat16:
            sorted_attention = sorted_attention.float()
        sorted_attention = sorted_attention.numpy()
        os.makedirs("out/fig/attn", exist_ok=True)
        np.save("out/fig/attn/sorted_attention.npy", sorted_attention)

        # for layer, attention in enumerate(sorted_attention):
        #     fig, ax = plt.subplots(figsize=(20, 10))
        #     total_tokens = attention.shape[0]

        #     specific_tokens = [(total_tokens//4)-1 , (total_tokens//2)-1, (3*total_tokens//4)-1]
        #     specific_colors = ['red', 'green', 'blue']  # Colors for the specific lines

        #     for t, token_attention in enumerate(attention):
        #         token_attention = token_attention[:t + 1]
        #         color = plt.cm.cool(t / total_tokens)
        #         ax.plot(token_attention, color=color, linewidth=0.2)
            
        #     for t in specific_tokens:
        #         token_attention = attention[t, :t + 1]
        #         color_index = specific_tokens.index(t)
        #         ax.plot(token_attention, color=specific_colors[color_index], linewidth=1.5, label=f'Token {t+1}')  # t+1 for 1-based indexing in label

                    
        #     # specifically, draw 512, 1024 
        #     smappable = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=0, vmax=total_tokens))
        #     cbar = fig.colorbar(smappable, ax=ax, label="Token Index")
            
        #     indices_to_show = [0, total_tokens//4, total_tokens//2, 3*total_tokens//4, total_tokens-1]
        #     cbar.set_ticks(indices_to_show)
        #     cbar.set_ticklabels([f"Token {i}" for i in indices_to_show])

        #     ax.tick_params(axis='both', which='major', labelsize=20)
        #     ax.grid(linestyle="--", linewidth=0.5)
        #     plt.legend(loc='upper right', fontsize=17)
        #     ax.set_title(f"Layer_{layer}_Sorted_Attention", fontsize=20)
            
        #     for desired_x_lim in [63, 127, 255, 511, total_tokens-1]:
        #         os.makedirs(f"out/fig/attn/x_log_y_log/{desired_x_lim+1}", exist_ok=True)
        #         os.makedirs(f"out/fig/attn/x_linear_y_log/{desired_x_lim+1}", exist_ok=True)
        #         os.makedirs(f"out/fig/attn/x_linear_y_linear/{desired_x_lim+1}", exist_ok=True)
        #         ax.set_xscale("symlog")
        #         ax.set_yscale("log")
        #         ax.set_xlim(0, desired_x_lim)
        #         y_min = np.min([attention[i, :min(i+1, desired_x_lim)].min() for i in range(total_tokens)])
        #         ax.set_ylim(y_min, 1)
        #         plt.savefig(f"out/fig/attn/x_log_y_log/{desired_x_lim+1}/Layer_{layer}.png", bbox_inches="tight")
        #         ax.set_xscale("linear")
        #         plt.savefig(f"out/fig/attn/x_linear_y_log/{desired_x_lim+1}/Layer_{layer}.png", bbox_inches="tight")
        #         ax.set_yscale("linear")
        #         plt.savefig(f"out/fig/attn/x_linear_y_linear/{desired_x_lim+1}/Layer_{layer}.png", bbox_inches="tight")


        #     plt.close()
        pass







    # throughput = ThroughputMonitor(fabric, window_size=50)
    # total_t0 = time.perf_counter()
    # train_iter = iter(train_dataloader)
    # for state["iter_num"] in range(state["iter_num"], max_iters):
    #     iter_num = state["iter_num"]
    #     lr = get_lr(iter_num, warmup_iters, lr_decay_iters) if decay_lr else learning_rate
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = lr

    #     if iter_num % eval_interval == 0:
    #         t0 = time.perf_counter()
    #         val_loss = validate(fabric, model, val_dataloader, max_iters=eval_iters)
    #         train_loss = validate(fabric, model, train_dataloader, max_iters=eval_iters)
    #         t1 = time.perf_counter() - t0
    #         fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, train loss {train_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
    #         fabric.log_dict(metrics = {"eval/val_loss": val_loss.item(),
    #                                    "eval/time": t1 * 1000,
    #                                    "eval/train_loss": train_loss.item(),
    #                                    "eval/lr": lr,
    #                                    "step": iter_num,
    #                                    "eval/valtime": t1 * 1000}, step=iter_num//log_interval)
    #         fabric.barrier()
    #     if iter_num % save_interval == 0:
    #         checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
    #         save_checkpoint(fabric, state, checkpoint_path)

    #     # Accumulate gradients over multiple micro-batches
    #     iter_t0 = time.perf_counter()
    #     gradient_accumulation_steps = hparams["gradient_accumulation_steps"]
    #     for micro_step in range(gradient_accumulation_steps):

    #         is_accumulating = micro_step == gradient_accumulation_steps - 1
    #         with fabric.no_backward_sync(model, enabled=is_accumulating):
    #             input_ids, targets = next(train_iter)
    #             logits = model(input_ids)

    #             loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    #             fabric.backward(loss / gradient_accumulation_steps)

    #     optimizer.step()
    #     optimizer.zero_grad()
    #     if iter_num % log_interval == 0:
    #         loss_item = loss.item()  # expensive device-to-host synchronization
    #         t1 = time.perf_counter()
    #         throughput.update(
    #             time=t1 - total_t0,
    #             batches=iter_num,
    #             samples=iter_num * batch_size,
    #             lengths=iter_num * batch_size * model.max_seq_length,
    #         )
    #         t_used = t1 - total_t0
    #         est_time = t_used / (iter_num + 1) * max_iters - t_used
    #         fabric.log_dict(metrics = {"running/iter": iter_num,
    #                                    "running/loss": loss_item,
    #                                    "running/lr": lr,
    #                                     "running/remaining_time": est_time / 60. / 60.,
    #                                    "running/itertime": (t1 - iter_t0) * 1000,
    #                                    "step": iter_num}, step=iter_num//log_interval)
    #         fabric.print(
    #             f"iter {iter_num}: loss {loss_item:.4f}, iter time:"
    #             f" {(t1 - iter_t0) * 1000:.2f}ms, est. time remaining: {est_time / 60. / 60.:.2f}h"
    #         )

# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    val_iter = iter(val_dataloader)

    losses = torch.zeros(max_iters, device=fabric.device)
    for k in range(max_iters):
        input_ids, targets = next(val_iter)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits, targets, chunk_size=0)
    out = losses.mean()

    model.train()
    return out


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

def load_datasets(data_dir: Path, max_seq_length: int) -> Tuple["Dataset", "Dataset"]:
    train_data = Dataset(data_dir / "train.bin", max_seq_length)
    val_data = Dataset(data_dir / "val.bin", max_seq_length)
    return train_data, val_data

class Dataset(IterableDataset):
    def __init__(self, data_file: Path, max_seq_length: int):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.max_seq_length, (1,)).item()
            x = torch.from_numpy((data[i : i + self.max_seq_length]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.max_seq_length]).astype(np.int64))
            yield x, y

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
