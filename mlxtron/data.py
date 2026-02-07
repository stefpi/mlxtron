
from pathlib import Path
from typing import Union
from datasets import load_dataset
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn

from mlxtron.config.config import Config, DataLoader

class DataLoader():
    def __init__(self, config: DataLoader):
        self.config = config

        self.global_batch_size = self.config.micro_batch_size * self.config.grad_acc_steps * self.dp
        self.num_global_micro_batches = self.global_batch_size // self.config.micro_batch_size

        self.dataset = load_dataset(self.config.dataset_name, split=self.config.split, name=self.config.subset_name, streaming=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        if self.config.num_samples:
            self.dataset = self.dataset.select(range(min(self.config.num_samples, len(self.dataset))))
        
        self.tokenized_dataset = self.tokenize_dataset(self.dataset, "text", self.config.seq_length, self.config.num_proc)