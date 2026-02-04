from typing import Union
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True

MlxtronConfigs = Union[LlamaConfig]
