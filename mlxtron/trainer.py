import mlx.core as mx
import mlx.nn as nn

class DistributedTrainer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: mx.distributed.Group = None
    ):
        super.__init__()
        self.module = module

        if group and group.size() > 1:
            self.group = group
        