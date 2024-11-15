## Download [phi](https://arxiv.org/abs/2309.05463) weights

### Phi 2

Microsoft Research [released](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) Phi 2, which is a 2.7 billion parameter model trained on "textbook-quality" data with knowledge distillation from Phi 1.5. The model achieves sota results among base LLMs with less than 13B parameters and matches or outperforms models up to 25x larger on complex benchmarks, e.g. it achieves better performance compared to 25x larger Llama-2-70B model on multi-step reasoning tasks, i.e., coding and math. Phi 2 was trained on 1.4T tokens and has not undergone any RLHF alignment nor has it been instruct fine-tuned. Phi 2 shares the same architecture with Phi 1.5 and has context length of 2048 tokens.
The model weights are released under [*Microsoft Research license*](https://huggingface.co/microsoft/phi-2#license).

To download the model weights and convert them to the lit-gpt format, run

```bash
pip install huggingface_hub
python scripts/download.py --repo_id microsoft/phi-2 --from_safetensors True
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/microsoft/phi-2
```

> [!WARNING]
> Phi-2 used [dropout](https://huggingface.co/microsoft/phi-2/blob/cb2f453/config.json#L26) during training which we don't model, so training will not be equal.

Inference the model in instruct mode:

```bash
python chat/base.py --checkpoint_dir checkpoints/microsoft/phi-2
```
```text
>> Prompt: Write a detailed analogy between mathematics and a lighthouse.
>> Reply: Mathematics is like a lighthouse. Mathematics provides a method to guide us through the sometimes chaotic and confusing waters of life. It provides a structured approach to problems which can help us find our way and provide direction. Just as a lighthouse keeps watch over the sea, mathematics can provide us with the tools to try and make sense of the world. Furthermore, just as a lighthouse keeps a watchful eye on the horizon, mathematics can help us reach our goals by showing us the way.
```

> [!NOTE]
> In order to obtain appropriate answers, you may need to tweak the [input prompt](https://github.com/Lightning-AI/lit-gpt/blob/74b8df0c3f07fc31d9d1a49e870a1f7955329ad8/chat/base.py#L359). E.g. we found out that if using `"Instruct:{prompt}\nOutput:\n"` instead of `"Instruct:{prompt}\nOutput:"` the model generates longer answers in some cases.

Free generation mode:
```bash
python generate/base.py --prompt "Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?\nBob:" --checkpoint_dir checkpoints/microsoft/phi-2
```
which yields
```text
Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
Bob: Well, one possible reason could be stress. Have you been feeling overwhelmed lately?
Alice: Yes, I've been juggling multiple deadlines and it's been quite taxing.
Carol: Stress can definitely impact your ability to concentrate. Maybe you need
```

### Phi 1.5

A team at Microsoft Research has made available Phi 1.5, which is a 1.3 billion parameter model optimized for common sense reasoning in natural language, showing performance on par with models 5x its size, especially in grade-school mathematics and basic coding. This model retains characteristics of larger LLMs, and significant improvement was noted in reducing toxic and biased generations by avoiding web data. It's also worth highlighting that while this model performs well on language understanding and common sense reasoning tasks, it is a base model that has not undergone any supervised instruction finetuning or finetuning with RLHF.

The model was trained the same data sources (7B tokens) as its [phi-1](https://arxiv.org/abs/2306.11644) predecessor, which includes

- a Python code subset from [The Stack](https://arxiv.org/abs/2211.15533) v1.2
- Q&A texts from [StackOverflow](https://archive.org/download/stackexchange)
- code from DeepMind [code_contests](https://github.com/deepmind/code_contests)
- synthetic Python textbooks and exercises generated by [gpt-3.5-turbo-0301](https://platform.openai.com/docs/models/gpt-3-5)

In addition, to create phi-1.5, the authors included additional textbook-quality synthetic text (roughly 20B tokens) in natural language, which was created using the [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) approach.

The model weights are released under a [*Microsoft Research license*](https://huggingface.co/microsoft/phi-1_5/blob/main/README.md#license).

In order to use the phi-1.5 model checkpoint, which requires about 3 Gb of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id microsoft/phi-1_5

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/microsoft/phi-1_5
```

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/microsoft/phi-1_5
```