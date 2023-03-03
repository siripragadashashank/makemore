import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)


class BLModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, x, y):
        logits = self.token_embedding_table(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            probs = F.softmax(logits, dim=1)
            



