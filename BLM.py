import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BLModel(nn.Module):
    def __init__(self, vocab_len, n_embed, block_size, n_head=4, n_layer=4):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_len, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.block = nn.Sequential(*[Block(n_embed, n_head, block_size) for _ in range(n_layer)])
        self.ln_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_len)

    def forward(self, x, y=None):
        b, t = x.shape
        token_embeddings = self.token_embedding_table(x)
        position_embeddings = self.position_embedding_table(torch.arange(t, device=device))
        x = token_embeddings + position_embeddings
        x = self.block(x)
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
            idx_c = idx[:, -self.block_size:]
            logits, loss = self(idx_c)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)

        w = q @ k.transpose(-2, -1) * c**-0.5
        w = w.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        out = w @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, n_emb, n_head, block_size):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_emb, block_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


