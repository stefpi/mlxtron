from datasets import load_dataset, Dataset
import numpy as np
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.data as dx

SEQ_LENGTH = 512
BATCH_SIZE = 4

world = mx.distributed.init()
rank = world.rank()
world_size = world.size()

shard_pct = 100 / world_size
shard_start = int(shard_pct * rank)
shard_end = int(shard_start + shard_pct)

dataset = Dataset.from_file("/Users/stefpi/.cache/huggingface/datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-train-00000-of-00004.arrow")

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token

# def tokenize_sample(example):
#     tokens = tokenizer.encode(example["text"])
#     if len(tokens) > SEQ_LENGTH:
#         tokens = tokens[:SEQ_LENGTH]
#     else:
#         tokens = tokens + [tokenizer.eos_token_id] * (SEQ_LENGTH - len(tokens))
#     return tokens


# sample = tokenize_sample(dataset[0])

buffer = dx.buffer_from_vector([{"data": dataset[0]["text"].encode("utf-8")}])

# samples = [{"text": tokenize_sample(dataset[i])} for i in range(len(dataset))]
# buffer = dx.buffer_from_vector(samples)

print(buffer)

# def bytes_to_array(b):
#     return mx.array(np.frombuffer(b, dtype=np.int32))


# stream = (
#     buffer.to_stream()
#     .key_transform("data", bytes_to_array)
#     .batch(BATCH_SIZE, dim={"data": 0})
# )