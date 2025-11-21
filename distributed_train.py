import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from mlxtron.llama import Model

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
    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return dataset, tokenizer

def batch_iterate(dataset, tokenizer, batch_size, block_size):
    # Create a generator that yields batches of tokens
    batch = []
    for example in dataset:
        text = example["text"]
        # Tokenize
        tokens = tokenizer.encode(text)
        
        # Skip short sequences for simplicity or pad? 
        # For this simple script, we'll just concatenate and chunk, or just take sequences that fit.
        # Let's try a simple approach: truncate or pad to block_size.
        
        if len(tokens) > block_size:
            tokens = tokens[:block_size]
        else:
            # Pad with eos_token
            tokens = tokens + [tokenizer.eos_token_id] * (block_size - len(tokens))
            
        batch.append(tokens)
        
        if len(batch) == batch_size:
            yield mx.array(batch)
            batch = []

def loss_fn(model, x, y):
    logits = model(x)
    # We want to predict the next token, so targets are x shifted by 1.
    # However, the standard way is usually passing inputs and targets.
    # Here we passed x as both input and target source.
    # logits shape: [B, L, V]
    # y shape: [B, L]
    
    # Cross entropy expects logits and targets.
    # We need to mask padding if we care about correctness, but for a simple script we might ignore it or assume the loss handles it if we passed ignore_index (MLX cross_entropy doesn't support ignore_index directly in the same way as PyTorch usually, check docs or assume standard).
    # MLX cross_entropy: "Computes the cross entropy loss between logits and targets."
    
    return nn.losses.cross_entropy(logits, y).mean()

def main():
    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()

    mx.random.seed(SEED)
    # np.random.seed(SEED)

    dataset, tokenizer = load_data_and_tokenizer()
    
    print("Initializing model...")
    model = Model(**MODEL_ARGS)
    mx.eval(model.parameters())
    
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    
    # Compile the training step
    def step(model, x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        grads = nn.average_gradients(grads)
        optimizer.update(model, grads)
        return loss

    # Create a state for the optimizer to compile it too if needed, but MLX handles it.
    
    print("Starting training...")
    step_count = 0
    
    # We need to iterate carefully since it's streaming
    data_iter = batch_iterate(dataset, tokenizer, BATCH_SIZE, BLOCK_SIZE)
    
    try:
        for batch in data_iter:
            # Prepare inputs and targets
            # x: [B, L-1]
            # y: [B, L-1] (next token)
            
            # Actually, let's just take the batch as is [B, L]
            # and split it.
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            loss = step(model, inputs, targets)
            mx.eval(model.parameters(), optimizer.state)
            
            step_count += 1
            if step_count % 10 == 0:
                print(f"Step {step_count}: Loss = {loss.item():.4f}")
            
            if step_count >= 100: # Run for 100 steps for demonstration
                print("Stopping after 100 steps for demonstration.")
                break
                
    except KeyboardInterrupt:
        print("Training interrupted.")

if __name__ == "__main__":
    main()
