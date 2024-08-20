import json
import torch
import torch.nn as nn
from torch.nn import functional as F

# Load the tokenizer mapping
with open('tokenizer.json', 'r') as f:
    tokenizer_mapping = json.load(f)

stoi = tokenizer_mapping['stoi']
itos = tokenizer_mapping['itos']

# Tokenization functions
def encode_text(s):
    return [int(stoi[c]) for c in s]

def decode_text(l):
    return ''.join([itos[str(i)] for i in l])

# Load and process the dataset
def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text_tokens = encode_text(item['text'])
            audio_tokens = item['audio']
            data.append({
                'text': text_tokens,
                'audio': audio_tokens
            })
    return data

# Load the dataset
dataset = load_dataset('dataset.jsonl')

# generate a random number from 0 to 1000
import random
random_number = random.randint(0, 1000)

# set seed
#torch.manual_seed(random_number)

# Hyperparameters -my
batch_size = 8 # 64
block_size = 1150 # 1150
max_iters = 2000
eval_interval = 200
learning_rate = 1e-4 # 5e-4
device = 'mps'
eval_iters = 100
n_embd = 192 # 384
n_head = 6 # 6
n_layer = 6 # 6
dropout = 0.1

# Vocabulary size (including special tokens)
vocab_size = len(stoi)

# Split the dataset into train and validation
train_data = dataset[:int(0.9 * len(dataset))]
val_data = dataset[int(0.9 * len(dataset)):]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    indices = torch.randint(len(data), (batch_size,))
    
    batch = []
    for i in indices:
        item = data[i]
        text_tokens = [int(stoi['<text>'])] + item['text'] + [int(stoi['<text>'])]
        audio_tokens = [int(stoi['<audio>'])] + item['audio'] + [int(stoi['<audio>'])]
        
        # Pad or truncate to block_size
        if len(text_tokens) + len(audio_tokens) < block_size:
            padding = [int(stoi['<pad>'])] * (block_size - len(text_tokens) - len(audio_tokens))
            sequence = text_tokens + audio_tokens + padding
        else:
            sequence = (text_tokens + audio_tokens)[:block_size]
        
        batch.append(sequence)
    
    x = torch.tensor(batch, dtype=torch.long).to(device)
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = x[:, 0]
    
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
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
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

    def generate(self, idx, max_new_tokens, use_split_token=False):
        split_token = ""

        if use_split_token:
            split_token = ","

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
            
            # Decode and print the newly generated token
            new_token = decode_text([idx_next.item()])
            print(new_token, end=split_token, flush=True)
        return idx

def train(model):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save the model
            torch.save(model.state_dict(), f'model_{iter}.pth')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log the iteration
        print(f"step {iter} / {max_iters}: loss {loss.item():.4f}")

# Modify the training loop
model = GPTLanguageModel()
model.load_state_dict(torch.load("model_1999.pth", map_location=torch.device('mps')))
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
#train(model)

# Generate from the modeд
tokens = [int(stoi[token]) for token in ['<text>', 'H', 'e', 'y', '!', '<text>', '<audio>']]
context = torch.tensor([tokens], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000, use_split_token=True)[0].tolist()