import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)


class BLModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, x, y=None):
        logits = self.token_embedding_table(x)
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
