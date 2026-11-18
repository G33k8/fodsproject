import numpy as np


def sample_t(n, a, p_neg, rng):
    neg = rng.random(n) < p_neg
    u = rng.random(n)
    return np.where(neg, -a * (1 - u), a * u)  # neg: [-a,0), pos: [0,a)


def make_dataset(n, a=2.0, sigma=0.2, p_neg=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = sample_t(n, a, p_neg, rng)
    x = t**2 + rng.normal(0.0, sigma, n)
    return x, t


def make_splits(a=2.0, sigma=0.2, p_neg=0.5, seed=0):
    sizes = {"train": (5000, 0), "val": (1000, 1000), "test": (2000, 2000)}
    return {k: make_dataset(n, a, sigma, p_neg, seed + off) for k, (n, off) in sizes.items()}
