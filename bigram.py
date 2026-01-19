import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8   # maximum context length for predictions
max_iters = 3000  # number of training iterations
eval_interval = 300  # evaluate the loss every eval_interval
learning_rate = 1e-2  # learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use GPU if available
eval_iters = 200  # number of iterations for evaluation
# ----------

torch.manual_seed(1337)  # for reproducibility


# Load dataset
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
print(f"length of dataset in characters: {len(text)}")

# Get vocabulary
chars = sorted(list(set(text)))  # get the unique chars
vocab_size = len(chars)
print(f"vocab_size: {vocab_size}")

# Create a mapping from char to int / Tokenization
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join([itos[i] for i in l])

# Train / test splits
data = torch.tensor(encode(text), dtype=torch.long) # convert the text to a tensor of integers
n = int(0.9 * len(data)) # first 90% will be train, rest validation
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
  """Generates a small batch of data of inputs x and targets y."""
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))  # generates random offsets to choose random sequences
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)  # move to device (GPU or CPU)
  return x, y


# Evaluation function
@torch.no_grad() # this decorator disables gradient tracking inside the function
def estimate_loss():
  """Estimates the loss on train and val sets."""
  out = {}
  model.eval()  # set the model to evaluation mode (disables dropout, batchNorm, etc.)
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()  # set the model back to training mode
  return out

# Define the Bigram Language Model (super simple version of a language model)
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # Each token directly reads off the logits for the next token from a lookup table
    # logits: the raw scores for each character in the vocabulary (initialized randomly)
    # This table maps each character index to a vector of size 'vocab_size'
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    # idx and targets are both (B, T) tensor of integers (B = Batch (number of sequences), T = Time (sequence length))
    logits = self.token_embedding_table(idx) # Output shape: (B, T, C) where C = embedding_size

    if targets is None:
      loss = None
    else:
      # reshaping to use cross_entropy
      B, T, C = logits.shape
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      logits, loss = self(idx)   # self(idx) invokes the __call__ method, which triggers the forward pass
      logits = logits[:, -1, :]  # keep only the last time step (B, C)
      probs = F.softmax(logits, dim=-1)  # apply softmax, (B, C)
      idx_next = torch.multinomial(probs, num_samples=1)  # sample from the distribution, (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # concatenate (B, T+1)
    return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device) # move the model to the GPU if available


# Create a pyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
  # every once in a while, evaluate the loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # :.4f formats the float to 4 decimal places
  
  xb, yb = get_batch("train") # sample a batch of data

  logits, loss = m(xb, yb) # evaluate the loss
  optimizer.zero_grad(set_to_none=True) # clear the gradients (set_to_none=True is a performance optimization, it sets gradients to None instead of zero)
  loss.backward() # compute the gradients
  optimizer.step() # update the weights

context = torch.zeros((1, 1), dtype=torch.long, device=device) # starting context with a single zero token ("\n")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


