from dataclasses import dataclass, field
from typing import Any, Union
from typing_extensions import Self
from pathlib import Path
import json

@dataclass
class MambaConfig:

    n_embd: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    block_size: int = 1024

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)
