import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256   # maximum context length for predictions
max_iters = 5000  # number of training iterations
eval_interval = 500  # evaluate the loss every eval_interval
learning_rate = 3e-4  # learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use GPU if available
eval_iters = 200  # number of iterations for evaluation
n_embd = 384  # embedding dimension
n_layer = 6  # number of transformer blocks
n_head = 6  # number of attention heads per block
dropout = 0.2  # dropout rate
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


# Define a single head of self-attention
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix for masking future tokens
        # tril is not a parameter, so we register it as a buffer (to be saved and moved to device with the model, but not trained)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, 16) @ (B, 16, T) ----> (B, T, T)
        # C**-0.5 is for scaling to prevent large dot products

        # mask out the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        
        return out
    

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd) # num_heads * head_size = n_embd // projection layer to combine the outputs of all heads
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the outputs of all heads along the embedding dimension
        out = self.dropout(self.proj(out))  # project back to the embedding dimension
        return out


class FeedForward(nn.Module):
    """A simple feed-forward neural network."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # increase dimension to allow for more complex transformations
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # project back to original embedding dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization before self-attention
        self.ln2 = nn.LayerNorm(n_embd) # layer normalization before feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Define the Bigram Language Model (super simple version of a language model)
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # logits: the raw scores for each character in the vocabulary (initialized randomly)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # token embeddings
    self.positional_embedding_table = nn.Embedding(block_size, n_embd) # positional embeddings
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # stack of transformer blocks (* unpacks the list into arguments)    self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to project the embeddings to vocabulary size
    self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to project the embeddings to vocabulary size


  def forward(self, idx, targets=None):
    # idx and targets are both (B, T) tensor of integers (B = Batch (number of sequences), T = Time (sequence length))
    B, T = idx.shape
    
    tok_embd = self.token_embedding_table(idx) # Output shape: (B, T, C) where C = embedding_size
    pos_embd = self.positional_embedding_table(torch.arange(T, device=device)) # Output shape: (T, C)
    x = tok_embd + pos_embd # broadcasting positional embeddings to all batch elements
    # x = self.sa_heads(x) # Output shape: (B, T, C)
    # x = self.ffwd(x) # Output shape: (B, T, C)
    x = self.blocks(x) # Output shape: (B, T, C)
    logits = self.lm_head(x) # Output shape: (B, T, vocab_size)

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
      idx_cond = idx[:, -block_size:]  # crop context to the last block_size tokens
      logits, loss = self(idx_cond)   # self(idx) invokes the __call__ method, which triggers the forward pass
      logits = logits[:, -1, :]  # keep only the last time step (B, C)
      probs = F.softmax(logits, dim=-1)  # apply softmax, (B, C)
      idx_next = torch.multinomial(probs, num_samples=1)  # sample from the distribution, (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # concatenate (B, T+1)
    return idx


model = BigramLanguageModel()
m = model.to(device) # move the model to the GPU if available

# print the number of parameters in the model
print(f"number of parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} million")

# Create a pyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
  # every once in a while, evaluate the loss on train and val sets
  if iter % eval_interval == 0 or iter + 1 == max_iters:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # :.4f formats the float to 4 decimal places
  
  xb, yb = get_batch("train") # sample a batch of data

  logits, loss = m(xb, yb) # evaluate the loss
  optimizer.zero_grad(set_to_none=True) # clear the gradients (set_to_none=True is a performance optimization, it sets gradients to None instead of zero)
  loss.backward() # compute the gradients
  optimizer.step() # update the weights

context = torch.zeros((1, 1), dtype=torch.long, device=device) # starting context with a single zero token ("\n")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


