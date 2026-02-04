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

world = mx.distributed.init()
rank = world.rank()
world_size = world.size()
shard_percentage = (100 / world_size)

def load_data_and_tokenizer():
    # if (rank == 0): print("Loading dataset...", flush=True)
    shard_start = int(shard_percentage * rank)
    shard_end = int(shard_start + shard_percentage)
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[{shard_start}%:{shard_end}%](pct1_dropremainder)", streaming=False)
    
    # if (rank == 0): print("Loading tokenizer...", flush=True)
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
        for batch in data_iter:
            if (rank == 0):
                inputs = batch[0:2, :-1]
                targets = batch[0:2, 1:]
            elif (rank == 1):
                inputs = batch[2:, :-1]
                targets = batch[2:, 1:]
            
            loss = step(model, inputs, targets)
            mx.eval(loss, model.parameters())
            
            step_count += 1

            if step_count >= 100: # Run for 100 steps for demonstration
                break
                
    except KeyboardInterrupt:
        print("Training interrupted.", flush=True)

if __name__ == "__main__":
    main()
