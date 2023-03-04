import torch
import torch.nn as nn
import torch.nn.functional as F


# hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
# -----------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as fp:
    text = fp.read()

chars = sorted(list(set(text)))
vocab_len = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]


def get_batch(split):
    data_split = train if split == 'train' else val
    ix = torch.randint(0, len(data_split) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    