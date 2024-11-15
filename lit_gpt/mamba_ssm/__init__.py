__version__ = "1.1.1"

from lit_gpt.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from lit_gpt.mamba_ssm.modules.mamba_simple import Mamba
from lit_gpt.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
