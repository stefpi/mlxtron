from typing import Optional, List, Union
from dataclasses import dataclass

import mlx.core as mx

from mlxtron.config.models import MlxtronConfigs

@dataclass
class ModelArgs:
    model_config: MlxtronConfigs

@dataclass
class DataArgs:
    dataset: str

@dataclass
class TokenizerArgs:
    """Arguments related to the tokenizer"""

    tokenizer_name_or_path: Optional[str] = None

@dataclass
class TokensArgs:
    """Arguments related to the tokens, sequence, batch and steps of the training"""

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

@dataclass
class LRSchedulerArgs:
    """Arguments related to the learning rate scheduler

    lr_warmup_steps: number of steps to warmup the learning rate
    lr_warmup_style: linear or constant
    lr_decay_style: linear, cosine or 1-sqrt
    min_decay_lr: minimum learning rate after decay
    lr_decay_steps: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    lr_decay_starting_step: optional number of steps to decay the learning rate otherwise will default to lr_warmup_steps
    """

    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = None
    lr_decay_style: str = None
    lr_decay_steps: Optional[int] = None
    lr_decay_starting_step: Optional[int] = None
    min_decay_lr: float = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_warmup_style is None:
            self.lr_warmup_style = "linear"
        if self.lr_decay_style is None:
            self.lr_decay_style = "linear"
        if self.lr_decay_style not in ["linear", "cosine", "1-sqrt"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine', '1-sqrt'] and not {self.lr_decay_style}"
            )
        if self.min_decay_lr is None:
            self.min_decay_lr = self.learning_rate

@dataclass
class SGDOptimizerArgs:
    name: str = "sgd"


@dataclass
class AdamWOptimizerArgs:
    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool
    name: str = "adamW"


@dataclass
class OptimizerArgs:
    """Arguments related to the optimizer and learning rate"""

    optimizer_factory: Union[SGDOptimizerArgs, AdamWOptimizerArgs]
    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]
    accumulate_grad_in_fp32: bool
    learning_rate_scheduler: LRSchedulerArgs
    weight_decay_exclude_named_params: Optional[
        List[str]
    ] = None  # List of regex patterns to exclude parameters from weight decay

    def __post_init__(self):
        if self.weight_decay_exclude_named_params is None:
            self.weight_decay_exclude_named_params: List[str] = []


@dataclass
class ParallelismArgs:
    dp: int
    tp: int
    cp: int
    ep: int
    pp: int

@dataclass
class DataLoaderArgs:
    dp: int
    micro_batch_size: int
    seq_length: int
    dataset_name: str
    tokenizer_name: str
    num_workers: int
    num_proc: int
    grad_acc_steps: int
    group: mx.distributed.Group
    subset_name: str = None
    split: str = "train"
    num_samples: int = None

@dataclass
class Config:
    """main config"""

    model: ModelArgs
    data: DataLoaderArgs
    parallelism: ParallelismArgs