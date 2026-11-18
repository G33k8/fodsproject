import copy
import random

import numpy as np
import torch

from models import MDN, MSENet, mdn_nll, to_tensor
from data import make_splits


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def mse_loss(model, x, t):
    return ((model(x) - t) ** 2).mean()


def nll_loss(model, x, t):
    return mdn_nll(*model(x), t)


def train(model, loss_fn, train, val, epochs=3000, lr=1e-3, batch_size=256,
          patience=200, clip=5.0, seed=0):
    xt, tt = to_tensor(train[0]), torch.as_tensor(train[1], dtype=torch.float32)
    xv, tv = to_tensor(val[0]), torch.as_tensor(val[1], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    h = {"train": [], "val": [], "best_epoch": 0, "best_val": float("inf"), "diverged": False}
    best_state, since = copy.deepcopy(model.state_dict()), 0
    n = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model, xt[idx], tt[idx])
            if not torch.isfinite(loss):
                h["diverged"] = True
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item() * len(idx)
        if h["diverged"]:
            break
        with torch.no_grad():
            v = loss_fn(model, xv, tv).item()
        h["train"].append(tot / n)
        h["val"].append(v)
        if not np.isfinite(v):
            h["diverged"] = True
            break
        if v < h["best_val"]:
            h["best_val"], h["best_epoch"], since = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return h


def fit_model(kind, splits, seed=0, M=3, var_floor=1e-4, a=2.0, **train_kw):
    set_seed(seed)
    if kind == "mse":
        model, loss_fn = MSENet(), mse_loss
    else:
        model, loss_fn = MDN(M=1 if kind == "gauss" else M, var_floor=var_floor, a=a), nll_loss
    h = train(model, loss_fn, splits["train"], splits["val"], seed=seed, **train_kw)
    return model, h
