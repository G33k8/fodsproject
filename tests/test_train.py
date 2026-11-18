import torch
from data import make_splits
from models import MSENet, to_tensor
from train import mse_loss, nll_loss, train, fit_model


def test_mse_loss_no_broadcast():
    m = MSENet(); x = torch.randn(8, 1); t = torch.randn(8)
    assert torch.isclose(mse_loss(m, x, t), ((m(x) - t) ** 2).mean())

def test_loss_decreases_and_best_restored():
    s = make_splits(seed=0)
    model, h = fit_model("mdn", s, seed=0, epochs=30, patience=30)
    assert h["val"][-1] < h["val"][0]
    x, t = s["val"]
    assert abs(nll_loss(model, to_tensor(x), torch.as_tensor(t, dtype=torch.float32)).item() - h["best_val"]) < 1e-5

def test_nan_loss_sets_diverged():
    s = make_splits(seed=0)
    h = train(MSENet(), lambda m, x, t: m(x).sum() * float("nan"), s["train"], s["val"], epochs=5)
    assert h["diverged"] is True

def test_same_seed_same_weights():
    s = make_splits(seed=0)
    a, _ = fit_model("gauss", s, seed=1, epochs=3)
    b, _ = fit_model("gauss", s, seed=1, epochs=3)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
