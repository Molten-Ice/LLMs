# %%
# Plan:
# 1. Follow andrej tutorial to code a basic transformer
# 3. Modify it to make multiple heads work in one.

# Finetuning
# 1. Getting gemini flash to generate a dataset.
# 2. Retrain on the new dataset to finetune it.
# 3. Add LoRA in as well

# %%
import torch

# %%
with open("asimov.txt", "r") as f:
    text = f.read()

print(f'length of text: {len(text):,}')

# %%
print(text[:1000])

# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'vocab size: {vocab_size}')
print(''.join(chars))

# %%
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Hari Seldon"))
print(decode(encode("Hari Seldon")))

# %%
import torch
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape)
print(data[:100])

# %%
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(f'len train_data: {len(train_data):,} | len val_data: {len(val_data):,}')

# %%
block_size = 8
train_data[:block_size+1]

# %%
x = train_data[:block_size+1]
y = train_data[1:block_size+2]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'input {context} has target {target} | "{decode(context.tolist())}" -> "{decode([target.tolist()])}"')


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# device = 'cpu'

# %%
torch.manual_seed(3)
batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == 'train' else val_data
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
    x, y = x.to(device), y.to(device)
    return x, y
@torch.no_grad()
def estimate_loss(model, eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
xb, yb = get_batch('train')
print(xb.shape)
print(yb.shape)

print(xb)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'"{decode(context.tolist())}" -> "{decode([target.tolist()])}"')


# %%


# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # Wants (B, C, T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(idx, max_new_tokens=300)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

batch_size = 32
for i in range(3000):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    if i % 100 == 0:
        losses = estimate_loss(model)
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # print(f'{i}: {loss.item():.4f}')

# %%
idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(idx, max_new_tokens=300)[0].tolist()))

# %%
torch.manual_seed(3)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# %%
xbow = torch.zeros((B, T, C))
print(xbow.shape)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = xprev.mean(dim=0)
   

# %%
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
# wei is (T, T) but pytorch broadcasts it to (B, T, T)
# First column is calculated individually! Second  and third columns are normal dot product
xbow = wei @ x # (B, T, T) @ (B, T, C) -> (B, T, C)

# %%
# Use softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ x
print(out.shape)
print(out)

# %%
torch.manual_seed(3)
B, T, C = 4, 8, 32 # batch, time, channels
x = torch.randn(B, T, C)

# Single Self-attention head
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# (B, T, C)
k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)
# (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
wei = q @ k.transpose(-2, -1) * head_size**-0.5

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
# output = wei @ x

v = value(x)
out = wei @ v
# print(out.shape)
out.shape



# %%
block_size = 8

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out


# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 2500
eval_interval = 500
learning_rate = 1e-3 # 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('asimov.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key_query_values = nn.Linear(n_embd, head_size*3, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k, q, v = x.split(self.head_size, dim=-1)
        print(f'k.shape: {k.shape}, q.shape: {q.shape}, v.shape: {v.shape}')

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        print(f'wei.shape: {wei.shape}')
        B,T,C = x.shape
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # wei = self.dropout(wei)

        out = wei @ v
        print(f'out.shape: {out.shape}')
        assert False
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.all_key_query_values = nn.Linear(n_embd, head_size*num_heads*3, bias=False)

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        all_key_query_values = self.all_key_query_values(x)
        outputs = []
        for i, key_query_values in enumerate(all_key_query_values.split(self.head_size*3, dim=-1)):
            outputs.append(self.heads[i](key_query_values))
        out = torch.cat(outputs, dim=-1)
        out = self.proj(out)
        # out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(f'{sum(p.numel() for p in m.parameters())/1e6:.2f} Million parameters')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# step 2499: train loss 1.1224, val loss 1.3344

# %%
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# %%



