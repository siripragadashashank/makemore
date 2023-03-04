import torch
from BLM import BLModel


def get_batch(split):
    data_split = train if split == 'train' else val
    ix = torch.randint(0, len(data_split) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _x, _y = get_batch(split)
            _logits, _loss = model(_x, _y)
            losses[k] = _loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model():
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    for i in range(max_iters):
        if i % eval_interval == 0:
            _out = estimate_loss(m)
            print(f"step {i}: train loss {_out['train']:.4f} val loss {_out['val']:.4f}")
        optimizer.zero_grad()
        x, y = get_batch('train')
        logits, loss = m(x, y)
        loss.backward()
        optimizer.step()


def generate():
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    gens = decode(m.generate(idx, max_tokens=300)[0].tolist())
    return gens


def encode(string):
    encoded = [stoi[char] for char in string]
    return encoded


def decode(indices):
    return "".join([itos[i] for i in indices])


if __name__ == '__main__':

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

    encoded_text = encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)
    n = int(0.9 * len(data))
    train = data[:n]
    val = data[n:]

    m = BLModel(vocab_len)
    m = m.to(device)

    train_model()
    output = generate()
    print(output)








