# try non streaming vs streaming HF dataset

from datasets import load_dataset
from transformers import AutoTokenizer

print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=False)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(next(iter(dataset)))