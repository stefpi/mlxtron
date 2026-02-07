import json
import argparse
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx

from datasets import load_dataset
from transformers import AutoTokenizer


def _get(config: dict, *keys, default=None):
    for k in keys:
        config = config.get(k, {}) if isinstance(config, dict) else {}
    return config if config != {} else default


def _shard_dataset(dataset_name: str, rank: int, world_size: int, split: str = "train", subset: str = None, num_samples: int = None):
    pct = 100 / world_size
    start = int(pct * rank)
    end = int(start + pct)
    split_spec = f"{split}[{start}%:{end}%]"
    kwargs = {"streaming": False}
    if subset:
        kwargs["name"] = subset
    dataset = load_dataset(dataset_name, split=split_spec, **kwargs)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def _tokenize_pad(example, tokenizer, seq_length: int, text_key: str = "text"):
    tokens = tokenizer.encode(example[text_key])
    if len(tokens) > seq_length:
        tokens = tokens[:seq_length]
    else:
        tokens = tokens + [tokenizer.eos_token_id] * (seq_length - len(tokens))
    return tokens

def _batch_to_stream(batch, micro_batch_size, transform):
    buffer = dx.buffer_from_vector(batch)
    return (
        buffer
        .to_stream()
        .key_transform("text", lambda x: transform(x))
        .batch(micro_batch_size)
        .prefetch(prefetch_size=8, num_threads=4)
    )


def _batch_stream(dataset, tokenizer, seq_length: int, per_rank_batch_size: int, micro_batch_size: int, text_key: str = "text"):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == per_rank_batch_size:
            return _batch_to_stream(batch, micro_batch_size, partial(_tokenize_pad, tokenizer=tokenizer, seq_length=seq_length, text_key=text_key))
    if batch:
        return _batch_to_stream(batch, micro_batch_size, partial(_tokenize_pad, tokenizer=tokenizer, seq_length=seq_length, text_key=text_key))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Distributed Args
    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()

    # Model Args
    model_config = config["model"]
    num_layers = model_config["num_layers"]
    vocab_size = model_config["vocab_size"]
    dims = model_config["dims"]
    mlp_dims = model_config["mlp_dims"]
    num_heads = model_config["num_heads"]

    # Data Args
    data_config = config["data"]
    micro_batch_size = data_config["micro_batch_size"]
    seq_length = data_config["seq_length"]
    dataset_name = data_config["dataset_name"]
    tokenizer_name = data_config["tokenizer_name"]
    grad_acc_steps = data_config.get("grad_acc_steps", 1)
    subset = data_config.get("subset_name")
    data_split = data_config.get("split", "train")
    num_samples = data_config.get("num_samples")
    text_key = "text"
    train_steps = 100

    # Parallelism Args
    parallelism_config = config["parallelism"]
    dp = parallelism_config["dp"]
    tp = parallelism_config["tp"]
    cp = parallelism_config["cp"]
    ep = parallelism_config["ep"]
    pp = parallelism_config["pp"]

    per_rank_batch_size = micro_batch_size * grad_acc_steps

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # shard dataset for rank into hugging face cache system
    dataset = _shard_dataset(dataset_name, rank, world_size, split=data_split, subset=subset, num_samples=num_samples)
    # bring rank batch size into memory using MLX data stream and tokenize
    batch_iter = _batch_stream(dataset, tokenizer, seq_length, per_rank_batch_size, micro_batch_size, text_key)
    # iterate over data stream
    step = 0
    for batch in batch_iter:
        if step >= train_steps:
            break
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        step += 1
