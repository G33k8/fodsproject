import numpy as np
import torch
from scipy.stats import norm
from models import MDN, MSENet, mdn_log_prob, mdn_sample, mdn_branch_prob


def test_mdn_shapes_and_constraints():
    log_pi, mu, var = MDN(M=3)(torch.linspace(-0.5, 4.5, 7)[:, None])
    assert log_pi.shape == mu.shape == var.shape == (7, 3)
    torch.testing.assert_close(log_pi.exp().sum(1), torch.ones(7))
    assert (var >= 1e-4).all()

def test_mean_bias_spread():
    torch.testing.assert_close(MDN(M=3, a=2.0).head.bias[3:6], torch.tensor([-2.0, 0.0, 2.0]))

def test_log_prob_matches_bruteforce():
    log_pi = torch.log(torch.tensor([[0.3, 0.7]]))
    mu, var, t = torch.tensor([[-1.0, 1.0]]), torch.tensor([[0.04, 0.09]]), torch.tensor([0.8])
    expected = np.log(0.3 * norm.pdf(0.8, -1, 0.2) + 0.7 * norm.pdf(0.8, 1, 0.3))
    assert abs(mdn_log_prob(log_pi, mu, var, t).item() - expected) < 1e-5

def test_log_prob_stable_far_from_all_components():
    log_pi = torch.log(torch.tensor([[0.5, 0.5]]))
    mu, var = torch.tensor([[0.0, 1.0]]), torch.tensor([[1e-4, 1e-4]])
    lp = mdn_log_prob(log_pi, mu, var, torch.tensor([100.0]))
    assert torch.isfinite(lp).all()

def test_branch_prob_matches_monte_carlo():
    log_pi = torch.log(torch.tensor([[0.2, 0.8]]))
    mu, var = torch.tensor([[-1.0, 0.3]]), torch.tensor([[0.25, 0.5]])
    g = torch.Generator().manual_seed(0)
    s = torch.stack([mdn_sample(log_pi, mu, var, g) for _ in range(20000)])
    assert abs((s > 0).float().mean() - mdn_branch_prob(log_pi, mu, var)[0]) < 0.01

def test_mse_net_output_is_1d():
    assert MSENet()(torch.zeros(5, 1)).shape == (5,)
