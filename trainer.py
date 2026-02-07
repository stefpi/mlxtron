import json
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx
from mlxtron import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

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
    # model_config = config["model"]
    # num_layers = model_config["num_layers"]
    # vocab_size = model_config["vocab_size"]
    # dims = model_config["dims"]
    # mlp_dims = model_config["mlp_dims"]
    # num_heads = model_config["num_heads"]

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
    # parallelism_config = config["parallelism"]
    # dp = parallelism_config["dp"]
    # tp = parallelism_config["tp"]
    # cp = parallelism_config["cp"]
    # ep = parallelism_config["ep"]
    # pp = parallelism_config["pp"]

    batch_iter = DataLoader(config["data"])

    step = 0
    for batch in batch_iter:
        if step >= train_steps:
            break
        
        print(batch.size)

        step += 1
