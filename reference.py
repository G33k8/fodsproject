import numpy as np
from scipy.integrate import trapezoid  # same rule as np.trapz (removed in NumPy 2.x)


def t_grid(a=2.0, n_grid=2001):
    return np.linspace(-a, a, n_grid)


def _logprior(t, a, p_neg):
    return np.log(np.where(t < 0, p_neg, 1 - p_neg) / a)


def _unnorm_log(x, t, a, sigma, p_neg):
    return _logprior(t, a, p_neg) - (x - t**2) ** 2 / (2 * sigma**2)


def posterior(x, a=2.0, sigma=0.2, p_neg=0.5, n_grid=2001):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = t_grid(a, n_grid)
    lp = _unnorm_log(x[:, None], t[None, :], a, sigma, p_neg)
    lp = lp - lp.max(axis=1, keepdims=True)  # log-space normalisation: no underflow
    p = np.exp(lp)
    return t, p / trapezoid(p, t, axis=1)[:, None]


def branch_prob(x, a=2.0, sigma=0.2, p_neg=0.5):
    t, p = posterior(x, a, sigma, p_neg)
    m = t >= 0
    return trapezoid(p[:, m], t[m], axis=1)


def posterior_mean(x, a=2.0, sigma=0.2, p_neg=0.5):
    t, p = posterior(x, a, sigma, p_neg)
    return trapezoid(p * t, t, axis=1)


def log_density_at(x, t, a=2.0, sigma=0.2, p_neg=0.5):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    g = t_grid(a)
    lg = _unnorm_log(x[:, None], g[None, :], a, sigma, p_neg)
    m = lg.max(axis=1)
    logz = m + np.log(trapezoid(np.exp(lg - m[:, None]), g, axis=1))
    out = _unnorm_log(x, t, a, sigma, p_neg) - logz
    return np.where(np.abs(t) > a, -np.inf, out)
