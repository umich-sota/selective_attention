# Selective attention
The code is modified from lit-gpt.
```bibtex
@misc{lit-gpt-2023,
  author       = {Lightning AI},
  title        = {Lit-GPT},
  howpublished = {\url{https://github.com/Lightning-AI/lit-gpt}},
  year         = {2023},
}
```

# Our paper
## Paper Abstract
The attention mechanism within the transformer architecture enables the model to weigh and combine tokens based on their relevance to the query. While self-attention has enjoyed major success, it notably treats all queries $q$ in the same way by applying the mapping $V^\top\text{softmax}(Kq)$, where $V,K$ are the value and key embeddings respectively. In this work, we argue that this uniform treatment hinders the ability to control contextual \xuechen{sparsity} and relevance. As a solution, we introduce the ``\emph{Selective Self-Attention}'' (SSA) layer that augments the softmax nonlinearity with a principled temperature scaling strategy. By controlling temperature, SSA adapts the contextual \xuechen{sparsity} of the attention map to the query embedding and its position in the context window. Through theory and experiments, we demonstrate that this alleviates attention dilution, aids the optimization process, and enhances the model's ability to control softmax spikiness of individual queries. We also incorporate temperature scaling for value embeddings and show that it boosts the model's ability to suppress irrelevant/noisy tokens. Notably, SSA is a lightweight method which introduces less than 0.5\% new parameters through a weight-sharing strategy and can be fine-tuned on existing LLMs. Extensive empirical evaluations demonstrate that SSA-equipped models achieve a noticeable and consistent accuracy improvement on language modeling benchmarks. 

Read our paper for more information about the setup. If you use our method, please cite us using
```bibtex
@inproceedings{
zhang2024selective,
title={Selective Attention: Enhancing Transformer through Principled Context Control},
author={Xuechen Zhang and Xiangyu Chang and Mingchen Li and Amit Roy-Chowdhury and Jiasi Chen and Samet Oymak},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=QbqLcwMXfF}
}
'''

