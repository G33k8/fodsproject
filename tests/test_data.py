import numpy as np
from data import make_dataset, make_splits


def test_shapes_and_range():
    x, t = make_dataset(5000, a=2.0, sigma=0.2, seed=0)
    assert x.shape == t.shape == (5000,)
    assert np.all(np.abs(t) <= 2.0)

def test_symmetric_negative_fraction():
    _, t = make_dataset(5000, p_neg=0.5, seed=0)
    assert abs((t < 0).mean() - 0.5) < 0.03

def test_asymmetric_negative_fraction():
    _, t = make_dataset(5000, p_neg=0.2, seed=0)
    assert abs((t < 0).mean() - 0.2) < 0.02

def test_noise_std_is_sigma():
    x, t = make_dataset(5000, sigma=0.2, seed=0)
    assert abs((x - t**2).std() - 0.2) < 0.01

def test_same_seed_same_data():
    assert np.array_equal(make_dataset(100, seed=3)[0], make_dataset(100, seed=3)[0])

def test_splits_differ():
    s = make_splits(seed=0)
    assert {k: len(v[0]) for k, v in s.items()} == {"train": 5000, "val": 1000, "test": 2000}
    assert not np.allclose(s["train"][0][:1000], s["test"][0][:1000])
    assert not np.allclose(s["val"][0], s["test"][0][:1000])
