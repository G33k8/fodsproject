import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class MSENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = Body()
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class MDN(nn.Module):
    def __init__(self, M=3, var_floor=1e-4, a=2.0):
        super().__init__()
        self.M, self.var_floor = M, var_floor
        self.body = Body()
        self.head = nn.Linear(64, 3 * M)  # [logits(M), mu(M), raw_var(M)]
        with torch.no_grad():
            self.head.bias[M:2 * M] = torch.linspace(-a, a, M)  # spread means against collapse

    def forward(self, x):
        logits, mu, raw = self.head(self.body(x)).split(self.M, dim=1)
        return F.log_softmax(logits, dim=1), mu, F.softplus(raw) + self.var_floor


def mdn_log_prob(log_pi, mu, var, t):
    z = log_pi - 0.5 * (torch.log(2 * math.pi * var) + (t[:, None] - mu) ** 2 / var)
    return torch.logsumexp(z, dim=1)


def mdn_nll(log_pi, mu, var, t):
    # PRD sum divided by N, so the value does not depend on batch size
    return -mdn_log_prob(log_pi, mu, var, t).mean()


def mdn_mean(log_pi, mu):
    return (log_pi.exp() * mu).sum(1)


def mdn_branch_prob(log_pi, mu, var):
    return (log_pi.exp() * torch.distributions.Normal(0.0, 1.0).cdf(mu / var.sqrt())).sum(1)


def mdn_sample(log_pi, mu, var, generator):
    k = torch.multinomial(log_pi.exp(), 1, generator=generator)
    m, v = mu.gather(1, k)[:, 0], var.gather(1, k)[:, 0]
    return m + v.sqrt() * torch.randn(m.shape, generator=generator, dtype=m.dtype)


def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1, 1)
