import numpy as np
import pytest
from scipy.integrate import trapezoid

from data import make_splits
from evaluate import (mean_loglik, predict_mean, model_density, forward_residuals, branch_error)
from models import MDN
from train import fit_model


@pytest.fixture(scope="module")
def splits():
    return make_splits(seed=0)


@pytest.fixture(scope="module")
def trained(splits):
    return {k: fit_model(k, splits, seed=0)[0] for k in ("mse", "gauss", "mdn")}


@pytest.mark.slow
def test_mdn_beats_gaussian_loglik(trained, splits):
    x, t = splits["test"]
    assert mean_loglik(trained["mdn"], x, t) > mean_loglik(trained["gauss"], x, t) + 0.3

@pytest.mark.slow
def test_mdn_branch_prob_within_005(trained):
    assert branch_error(trained["mdn"], 2.0, 0.2, 0.5) < 0.05

@pytest.mark.slow
def test_residual_spread_close_to_sigma(trained, splits):
    r = forward_residuals(trained["mdn"], splits["test"][0])
    assert abs(r.std() - 0.2) < 0.05

@pytest.mark.slow
def test_mse_net_predicts_near_zero_between_branches(trained):
    assert np.abs(predict_mean(trained["mse"], np.linspace(1, 3.5, 20))).max() < 0.15

def test_model_density_integrates_to_one():
    m = MDN(M=3); t = np.linspace(-6, 6, 4001)
    assert abs(trapezoid(model_density(m, 1.0, t), t) - 1) < 1e-3

def test_fig_reference_writes_file(tmp_path):
    from evaluate import fig_reference
    p = fig_reference(tmp_path)
    assert p.exists() and p.stat().st_size > 10_000

def test_experiment_matrix_shape():
    from evaluate import EXPERIMENTS
    assert {k: len(v) for k, v in EXPERIMENTS.items()} == {"baseline": 3, "noise": 3, "prior": 3, "components": 4, "floor": 3}

def test_run_one_survives_divergence(monkeypatch):
    import json
    from evaluate import run_one
    monkeypatch.setattr("train.train", lambda *a, **k: {"train": [], "val": [float("nan")], "best_epoch": 0, "best_val": float("nan"), "diverged": True})
    row = run_one({"kind": "mdn", "sigma": 0.2, "p_neg": 0.5, "M": 3, "var_floor": 0.0}, seed=0)
    assert row["diverged"] is True
    json.dumps(row)

def test_run_experiments_resumes(tmp_path, monkeypatch):
    import evaluate
    calls = []
    monkeypatch.setattr(evaluate, "run_one", lambda cfg, seed: calls.append(1) or {**cfg, "seed": seed})
    p = tmp_path / "e.json"; evaluate.run_experiments(p); n = len(calls); evaluate.run_experiments(p)
    assert len(calls) == n
