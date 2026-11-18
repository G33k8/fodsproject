import numpy as np
from scipy.integrate import trapezoid
from reference import posterior, branch_prob, posterior_mean, log_density_at

XS = [-0.2, 0.1, 1.0, 3.0, 4.5]

def test_integrates_to_one():
    t, p = posterior(XS)
    np.testing.assert_allclose(trapezoid(p, t, axis=1), 1.0, atol=1e-6)

def test_two_peaks_at_x1():
    t, p = posterior(1.0)
    left, right = p[0][t < 0], p[0][t >= 0]
    assert abs(t[t < 0][left.argmax()] + 1) < 0.01 and abs(t[t >= 0][right.argmax()] - 1) < 0.01

def test_single_blob_at_negative_x():
    t, p = posterior(-0.2)
    assert abs(t[p[0].argmax()]) < 0.05

def test_far_x_is_finite():
    t, p = posterior([10.0, -1.0])
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(trapezoid(p, t, axis=1), 1.0, atol=1e-6)

def test_branch_prob_symmetric():
    np.testing.assert_allclose(branch_prob([0.5, 1.0, 3.0]), 0.5, atol=1e-3)

def test_branch_prob_asymmetric():
    assert abs(branch_prob(1.0, p_neg=0.2)[0] - 0.8) < 0.01

def test_posterior_mean_symmetric_is_zero():
    np.testing.assert_allclose(posterior_mean([0.5, 1.0, 3.0]), 0.0, atol=1e-6)

def test_log_density_at_matches_grid():
    t, p = posterior(1.0)
    assert abs(log_density_at(np.array([1.0]), np.array([t[1500]]))[0] - np.log(p[0, 1500])) < 1e-6
