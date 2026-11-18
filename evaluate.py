import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from scipy.integrate import trapezoid

import reference
from models import MDN, MSENet, mdn_log_prob, mdn_mean, mdn_branch_prob, mdn_sample, to_tensor

# ---------------------------------------------------------------- metrics

def _t(t):
    return torch.as_tensor(np.asarray(t), dtype=torch.float32)


@torch.no_grad()
def mean_loglik(model, x, t):
    return mdn_log_prob(*model(to_tensor(x)), _t(t)).mean().item()


@torch.no_grad()
def predict_mean(model, x):
    out = model(to_tensor(x))
    return (out if isinstance(model, MSENet) else mdn_mean(out[0], out[1])).numpy()


def test_mse(model, x, t):
    return float(((predict_mean(model, x) - np.asarray(t)) ** 2).mean())
test_mse.__test__ = False  # not a pytest test


@torch.no_grad()
def model_density(model, x, t):
    log_pi, mu, var = model(to_tensor([x]))
    tt = _t(t)[:, None]
    z = log_pi - 0.5 * (torch.log(2 * np.pi * var) + (tt - mu) ** 2 / var)
    return torch.logsumexp(z, dim=1).exp().numpy().astype(float)


@torch.no_grad()
def model_branch_prob(model, x):
    return mdn_branch_prob(*model(to_tensor(x))).numpy()


@torch.no_grad()
def forward_residuals(model, x, seed=0):
    """x - t_s^2 with one sample t_s ~ p(t|x) per test x. Marginally over the
    test set this is distributed as N(0, sigma^2) when the model is correct."""
    g = torch.Generator().manual_seed(seed)
    ts = mdn_sample(*model(to_tensor(x)), g).numpy().astype(float)
    return np.asarray(x) - ts**2


BRANCH_X = np.linspace(0.5, 3.5, 61)


def branch_error(model, a, sigma, p_neg):
    ref = reference.branch_prob(BRANCH_X, a, sigma, p_neg)
    return float(np.abs(model_branch_prob(model, BRANCH_X) - ref).max())
