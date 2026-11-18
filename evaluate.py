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

# ---------------------------------------------------------------- figures
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from data import make_splits
from train import fit_model

FIG_X = [-0.2, 0.1, 1.0, 3.0, 4.5]
A, SIGMA = 2.0, 0.2


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _panels(n):
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3), sharey=False)
    return fig, np.atleast_1d(axs)


def _sqrt_marks(ax, x):
    if x > 0:
        for s in (-1, 1):
            ax.axvline(s * np.sqrt(x), color="gray", ls=":", lw=1)


def fig_reference(out_dir):
    fig, axs = _panels(len(FIG_X))
    t, p = reference.posterior(FIG_X)
    for ax, x, row in zip(axs, FIG_X, p):
        ax.plot(t, row, color="k")
        _sqrt_marks(ax, x)
        ax.set_title(f"x = {x}")
        ax.set_xlabel("t")
    axs[0].set_ylabel("p(t | x)")
    return _save(fig, out_dir, "part1_reference.png")


def fig_mse(models, splits, out_dir):
    x, t = splits["train"]
    g = np.linspace(-0.5, 4.5, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, t, s=3, alpha=0.3, label="training data")
    ax.plot(g, predict_mean(models["mse"], g), "r", lw=2, label="MSE network")
    ax.plot(g, reference.posterior_mean(g), "k--", label="E[t | x] (reference)")
    gp = g[g > 0]
    ax.plot(gp, np.sqrt(gp), color="gray", ls=":", label=r"$\pm\sqrt{x}$")
    ax.plot(gp, -np.sqrt(gp), color="gray", ls=":")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.legend(fontsize=8)
    return _save(fig, out_dir, "part2_mse.png")


def fig_overlays(models, out_dir):
    fig, axs = _panels(len(FIG_X))
    t = reference.t_grid()
    _, ref = reference.posterior(FIG_X)
    for ax, x, row in zip(axs, FIG_X, ref):
        ax.plot(t, row, "k", lw=2, label="reference")
        ax.plot(t, model_density(models["gauss"], x, t), "b--", label="single Gaussian")
        ax.plot(t, model_density(models["mdn"], x, t), "r-", lw=1.2, label="MDN")
        ax.set_title(f"x = {x}"); ax.set_xlabel("t")
    axs[0].set_ylabel("density"); axs[0].legend(fontsize=7)
    return _save(fig, out_dir, "part4_overlays.png")


def fig_branch_prob(models, out_dir):
    g = np.linspace(-0.5, 4.5, 201)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g, reference.branch_prob(g), "k", lw=2, label="reference")
    ax.plot(g, model_branch_prob(models["mdn"], g), "r--", label="MDN")
    ax.axhline(0.5, color="gray", ls=":")
    ax.axvspan(BRANCH_X[0], BRANCH_X[-1], color="C0", alpha=0.1, label="evaluation range")
    ax.set_xlabel("x"); ax.set_ylabel("P(t > 0 | x)"); ax.legend(fontsize=8)
    return _save(fig, out_dir, "part4_branch_prob.png")


def fig_residuals(models, splits, sigma, out_dir):
    x = splits["test"][0]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    z = np.linspace(-1.2, 1.2, 400)
    pdf = np.exp(-z**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    for ax, k, name in zip(axs, ("gauss", "mdn"), ("single Gaussian", "MDN")):
        r = forward_residuals(models[k], x)
        ax.hist(r, bins=80, range=(-1.2, 1.2), density=True, alpha=0.6, label=f"{name} (std {r.std():.2f})")
        ax.plot(z, pdf, "k", label=rf"$N(0,\sigma^2)$, $\sigma$={sigma}")
        ax.set_xlabel(r"$x - t^2$"); ax.legend(fontsize=7)
    return _save(fig, out_dir, "part4_residuals.png")


def table_baseline(models, splits):
    x, t = splits["test"]
    ref_ll = reference.log_density_at(x, t).mean()
    ref_mse = float(((reference.posterior_mean(x) - t) ** 2).mean())
    rows = [("Exact reference", f"{ref_ll:.3f}", f"{ref_mse:.3f}"),
            ("MSE network", "---", f"{test_mse(models['mse'], x, t):.3f}")]
    for k, n in (("gauss", "Single Gaussian"), ("mdn", "MDN")):
        rows.append((n, f"{mean_loglik(models[k], x, t):.3f}", f"{test_mse(models[k], x, t):.3f}"))
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    return ("\\begin{tabular}{lrr}\n\\toprule\nModel & Test mean log-lik. & Test MSE \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n")


def train_baselines(seed=0):
    s = make_splits(seed=0)
    return {k: fit_model(k, s, seed=seed)[0] for k in ("mse", "gauss", "mdn")}, s


def cmd_figures(fig_dir="figures", res_dir="results"):
    models, s = train_baselines()
    fig_reference(fig_dir); fig_mse(models, s, fig_dir); fig_overlays(models, fig_dir)
    fig_branch_prob(models, fig_dir); fig_residuals(models, s, SIGMA, fig_dir)
    Path(res_dir).mkdir(exist_ok=True)
    (Path(res_dir) / "table_baseline.tex").write_text(table_baseline(models, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["figures", "experiments"])
    a = ap.parse_args()
    if a.cmd == "figures":
        cmd_figures()
    else:
        cmd_experiments()


if __name__ == "__main__":
    main()
