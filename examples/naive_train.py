from profiler import Profiler
p = Profiler()
t = Profiler()

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlxtron.llama import Model

from datasets import load_dataset
from transformers import AutoTokenizer

# Constants
BATCH_SIZE = 4
BLOCK_SIZE = 512
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
SEED = 42

# Model Configuration (Small model for testing/demonstration)
MODEL_ARGS = {
    "num_layers": 4,
    "vocab_size": 50257,  # GPT-2 vocab size
    "dims": 256,
    "mlp_dims": 512,
    "num_heads": 4,
}

def load_data_and_tokenizer():
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return dataset, tokenizer

def batch_iterate(dataset, tokenizer, batch_size, block_size):
    # Create a generator that yields batches of tokens
    batch = []
    for example in dataset:
        text = example["text"]
        tokens = tokenizer.encode(text)

        # truncate or pad to block_size.
        if len(tokens) > block_size:
            tokens = tokens[:block_size]
        else:
            # Pad with eos_token
            tokens = tokens + [tokenizer.eos_token_id] * (block_size - len(tokens))
            
        batch.append(tokens)
        
        if len(batch) == batch_size:
            yield mx.array(batch)
            batch = []

def main():
    mx.random.seed(SEED)
    # np.random.seed(SEED)

    dataset, tokenizer = load_data_and_tokenizer()
    model = Model(**MODEL_ARGS)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)

    def loss_fn(x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y).mean()

    def step(model, x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(x,y)
        grads = nn.average_gradients(grads)
        optimizer.update(model, grads)
        return loss
    
    step_count = 0
    
    data_iter = batch_iterate(dataset, tokenizer, BATCH_SIZE, BLOCK_SIZE)

    try:
        t.start("total training time")
        for batch in data_iter:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            p.start("step and eval")
            loss = step(model, inputs, targets)
            mx.eval(loss, model.parameters())
            p.end()
            
            step_count += 1
            if step_count % 10 == 0:
                print(f"Step {step_count}: Loss = {loss.item():.4f}")
            
            if step_count >= 100: # Run for 100 steps for demonstration
                print("Stopping after 100 steps for demonstration.")
                break
        t.end()
                
    except KeyboardInterrupt:
        print("Training interrupted.")

if __name__ == "__main__":
    mx.reset_peak_memory()
    print(mx.get_peak_memory())
    main()
    print(mx.get_peak_memory())
    