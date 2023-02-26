import torch
import typing as ty
torch.manual_seed(42)


class Linear:
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.tensor) -> torch.tensor:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self) -> ty.List[torch.tensor]:
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # params trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # running mean and std
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.tensor) -> torch.tensor:
        if self.training:
            x_mean = x.mean(dim=0, keepdims=True)
            x_var = x.var(dim=0, keepdims=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        x_hat = (x-x_mean)/torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self) -> ty.List[torch.tensor]:
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x: torch.tensor) -> torch.tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embedding, embed_dim):
        self.weight = torch.randn((num_embedding, embed_dim))

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, x: torch.tensor) -> torch.tensor:
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    # @property
    # def layers(self):
    #     return self.layers

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        return params
