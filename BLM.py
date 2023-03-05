import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

n_embed = 32
vocab_len = 65
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# head_size = 16


class BLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_len, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4)
        self.lm_head = nn.Linear(n_embed, vocab_len)

    def forward(self, x, y=None):
        b, t = x.shape
        token_embeddings = self.token_embedding_table(x)
        position_embeddings = self.position_embedding_table(torch.arange(t, device=device))
        x = token_embeddings + position_embeddings
        x = self.sa_heads(x)
        logits = self.lm_head(x)
        if y is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            y = y.view(b*t)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_c = idx[:, -block_size:]
            logits, loss = self(idx_c)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):

        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)

        w = q @ k.transpose(-2, -1) * c**-0.5
        w = w.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)

        v = self.value(x)
        out = w @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)



