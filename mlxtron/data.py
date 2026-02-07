
from pathlib import Path
from typing import Union
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx
from mlxtron.config.config import Config, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

class DataLoader():
    def __init__(self, config: DataLoader):
        data_config = config["data"]

        self.group = config["group"]
        self.world_size = self.group.world_size()
        self.rank = self.group.rank()
        
        self.dataset_name = data_config["dataset_name"]
        tokenizer_name = data_config["tokenizer_name"]
        
        self.subset = data_config.get("subset_name")
        self.data_split = data_config.get("split", "train")
        self.num_samples = data_config.get("num_samples")

        self.text_key = "text"
        self.seq_length = data_config["seq_length"]
        self.micro_batch_size = data_config["micro_batch_size"]
        self.grad_acc_steps = data_config.get("grad_acc_steps", 1)
        self.per_rank_batch_size = self.micro_batch_size * self.grad_acc_steps

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # shard dataset for rank into hugging face cache system
        self.dataset = self._shard_dataset()

    def _shard_dataset(self):
        pct = 100 / self.world_size
        start = int(pct * self.rank)
        end = int(start + pct)
        split_spec = f"{self.split}[{start}%:{end}%]"
        kwargs = {"streaming": False}
        if self.subset:
            kwargs["name"] = self.subset
        dataset = load_dataset(self.dataset_name, split=split_spec, **kwargs)
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        return dataset
    
    def _tokenize_pad(self, example):
        tokens = self.tokenizer.encode(example[self.text_key])
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        else:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.seq_length - len(tokens))
        return tokens

    def _batch_to_stream(self, batch):
        buffer = dx.buffer_from_vector(batch)
        return (
            buffer
            .to_stream()
            .key_transform("text", lambda x: self._tokenize_pad(x))
            .batch(self.micro_batch_size)
            .prefetch(prefetch_size=8, num_threads=4)
        )

    # bring rank batch size into memory using MLX data stream and tokenize
    def __iter__(self):
        batch = []
        for example in self.dataset:
            batch.append(example)
            if len(batch) == self.per_rank_batch_size:
                yield self._batch_to_stream(batch)
        if batch:
            yield self._batch_to_stream(batch)
