import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import reference
import train as train_mod  # module import so tests can monkeypatch train_mod.fit_model
from data import make_splits
from models import MSENet, mdn_log_prob, mdn_mean, mdn_branch_prob, mdn_sample, to_tensor

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
    out = [p.expand(len(t), -1) for p in model(to_tensor([x]))]
    return mdn_log_prob(*out, _t(t)).exp().numpy().astype(float)


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
    return {k: train_mod.fit_model(k, s, seed=seed)[0] for k in ("mse", "gauss", "mdn")}, s


def cmd_figures(fig_dir="figures", res_dir="results"):
    models, s = train_baselines()
    fig_reference(fig_dir); fig_mse(models, s, fig_dir); fig_overlays(models, fig_dir)
    fig_branch_prob(models, fig_dir); fig_residuals(models, s, SIGMA, fig_dir)
    Path(res_dir).mkdir(exist_ok=True)
    (Path(res_dir) / "table_baseline_seed0.tex").write_text(table_baseline(models, s))


# ---------------------------------------------------------------- experiments

SEEDS = [0, 1, 2, 3, 4]
_D = dict(sigma=0.2, p_neg=0.5, M=3, var_floor=1e-4)
HARSH_SIGMA = 0.02  # peak std ~ sigma/(2 sqrt x) = 0.01 at x=1, i.e. variance ~ the 1e-4 floor


def _cfgs(kind="mdn", base=None, **vary):
    return [{"kind": kind, **_D, **(base or {}), k: v} for k, vals in vary.items() for v in vals]


EXPERIMENTS = {
    "baseline": [{"kind": k, **_D} for k in ("mse", "gauss", "mdn")],
    "noise": _cfgs(sigma=[0.05, 0.2, 0.5]),
    "prior": _cfgs(p_neg=[0.5, 0.2, 0.05]),
    "components": _cfgs(M=[1, 2, 3, 5]),
    "floor": _cfgs(var_floor=[0.0, 1e-4, 1e-2]),
    # harsher Part 5 settings: narrow peaks make the floor bind; very rare negative branch
    "harsh_floor": _cfgs(base={"sigma": HARSH_SIGMA}, var_floor=[0.0, 1e-4, 1e-2]),
    "harsh_prior": _cfgs(p_neg=[0.02, 0.01]),
}


def _f(v):
    return float(v) if v is not None and math.isfinite(v) else None


def run_one(cfg, seed):
    a = A
    s = make_splits(a, cfg["sigma"], cfg["p_neg"], seed=0)
    model, h = train_mod.fit_model(cfg["kind"], s, seed=seed, M=cfg["M"],
                                   var_floor=cfg["var_floor"], a=a)
    x, t = s["test"]
    row = {**cfg, "seed": seed, "best_epoch": h["best_epoch"], "diverged": bool(h["diverged"]),
           "test_loglik": None, "test_mse": None, "branch_error": None,
           "residual_std": None, "rare_branch_recovered": None}
    try:
        row["test_mse"] = _f(test_mse(model, x, t))
        if cfg["kind"] != "mse":
            row["test_loglik"] = _f(mean_loglik(model, x, t))
            row["branch_error"] = _f(branch_error(model, a, cfg["sigma"], cfg["p_neg"]))
            row["residual_std"] = _f(forward_residuals(model, x).std())
            mb = 1 - model_branch_prob(model, np.array([2.0]))[0]
            rb = 1 - reference.branch_prob(2.0, a, cfg["sigma"], cfg["p_neg"])[0]
            row["rare_branch_recovered"] = bool(mb > 0.5 * rb)
    except Exception:  # diverged weights can be NaN
        if not h["diverged"]:
            raise
    return row


def _key(r):
    return json.dumps([{k: r[k] for k in ("kind", "sigma", "p_neg", "M", "var_floor")}, r["seed"]], sort_keys=True)


def run_experiments(path=Path("results/experiments.json")):
    path = Path(path)
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {_key(r) for r in rows}
    for cfgs in EXPERIMENTS.values():
        for cfg in cfgs:
            for seed in SEEDS:
                if _key({**cfg, "seed": seed}) in done:
                    continue
                rows.append(run_one(cfg, seed))
                done.add(_key(rows[-1]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows, indent=1))
    return rows


def _ms(vals, fmt="{:.3f}"):
    v = [x for x in vals if x is not None]
    if not v:
        return "---"
    return (fmt + r" $\pm$ " + fmt).format(np.mean(v), np.std(v))


def summarize(rows):
    var = {"baseline": "kind", "noise": "sigma", "prior": "p_neg", "components": "M", "floor": "var_floor",
           "harsh_floor": "var_floor", "harsh_prior": "p_neg"}
    out = {}
    for name, cfgs in EXPERIMENTS.items():
        lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
                 f"{var[name].replace('_', ' ')} & log-lik. & MSE & branch err. & resid. std & rare branch & diverged \\\\", "\\midrule"]
        for cfg in cfgs:
            rs = [r for r in rows if all(r[k] == cfg[k] for k in cfg) and r["seed"] in SEEDS]
            rare = [r["rare_branch_recovered"] for r in rs if r.get("rare_branch_recovered") is not None]
            lines.append(" & ".join([
                str(cfg[var[name]]), _ms([r["test_loglik"] for r in rs]), _ms([r["test_mse"] for r in rs]),
                _ms([r["branch_error"] for r in rs]), _ms([r["residual_std"] for r in rs]),
                f"{sum(rare)}/{len(rare)}" if rare else "---",
                f"{sum(r['diverged'] for r in rs)}/{len(rs)}"]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        out[name] = "\n".join(lines)
    return out


def _fit(kind, **kw):
    cfg = {**_D, **kw}
    s = make_splits(A, cfg["sigma"], cfg["p_neg"], seed=0)
    m, h = train_mod.fit_model(kind, s, seed=0, M=cfg["M"], var_floor=cfg["var_floor"])
    return m, h, s


def fig_part5(out_dir):
    g = np.linspace(-0.5, 4.5, 201)
    t = reference.t_grid()
    # prior
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, pn in enumerate((0.5, 0.2, 0.05)):
        m, _, _ = _fit("mdn", p_neg=pn)
        axs[0].plot(g, reference.branch_prob(g, p_neg=pn), color=f"C{i}", lw=2, label=f"ref, P(t<0)={pn}")
        axs[0].plot(g, model_branch_prob(m, g), color=f"C{i}", ls="--")
        axs[1].plot(t, reference.posterior(1.0, p_neg=pn)[1][0], color=f"C{i}", lw=2)
        axs[1].plot(t, model_density(m, 1.0, t), color=f"C{i}", ls="--")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("P(t > 0 | x)"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("t"); axs[1].set_title("density at x = 1 (solid ref, dashed MDN)")
    _save(fig, out_dir, "part5_prior.png")
    # components
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, x in zip(axs, (1.0, 3.0)):
        ax.plot(t, reference.posterior(x)[1][0], "k", lw=2, label="reference")
        for i, M in enumerate((1, 2, 3, 5)):
            m, _, _ = _fit("mdn", M=M)
            ax.plot(t, model_density(m, x, t), color=f"C{i}", ls="--", label=f"M={M}")
        ax.set_title(f"x = {x}"); ax.set_xlabel("t")
    axs[0].legend(fontsize=7)
    _save(fig, out_dir, "part5_components.png")
    # floor
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[1].plot(t, reference.posterior(1.0)[1][0], "k", lw=2, label="reference")
    for i, fl in enumerate((0.0, 1e-4, 1e-2)):
        m, h, _ = _fit("mdn", var_floor=fl)
        axs[0].plot(h["val"], color=f"C{i}", label=f"floor={fl}")
        if h["diverged"]:
            axs[0].plot(len(h["val"]), h["val"][-1] if h["val"] and np.isfinite(h["val"][-1]) else 0, "x", color=f"C{i}", ms=10)
        with np.errstate(all="ignore"):
            d = model_density(m, 1.0, t)
        axs[1].plot(t, d, color=f"C{i}", ls="--", label=f"floor={fl}")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("validation NLL"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("t"); axs[1].legend(fontsize=7); axs[1].set_title("density at x = 1")
    _save(fig, out_dir, "part5_floor.png")
    # noise
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, sg in zip(axs, (0.05, 0.2, 0.5)):
        m, _, _ = _fit("mdn", sigma=sg)
        ax.plot(t, reference.posterior(1.0, sigma=sg)[1][0], "k", lw=2, label="reference")
        ax.plot(t, model_density(m, 1.0, t), "r--", label="MDN")
        ax.set_title(rf"$\sigma$ = {sg}, x = 1"); ax.set_xlabel("t")
    axs[0].legend(fontsize=7)
    _save(fig, out_dir, "part5_noise.png")
    # harsh: floor at sigma=0.02, and very rare negative branch
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(t, reference.posterior(1.0, sigma=HARSH_SIGMA)[1][0], "k", lw=2, label="reference")
    for i, fl in enumerate((0.0, 1e-4, 1e-2)):
        m, _, _ = _fit("mdn", sigma=HARSH_SIGMA, var_floor=fl)
        axs[0].plot(t, model_density(m, 1.0, t), color=f"C{i}", ls="--", label=f"floor={fl}")
    axs[0].set_xlim(-1.2, 1.2); axs[0].set_xlabel("t"); axs[0].legend(fontsize=7)
    axs[0].set_title(rf"$\sigma$ = {HARSH_SIGMA}, x = 1")
    for i, pn in enumerate((0.02, 0.01)):
        m, _, _ = _fit("mdn", p_neg=pn)
        axs[1].plot(t, reference.posterior(2.0, p_neg=pn)[1][0], color=f"C{i}", lw=2, label=f"ref, P(t<0)={pn}")
        axs[1].plot(t, model_density(m, 2.0, t), color=f"C{i}", ls="--", label=f"MDN, P(t<0)={pn}")
    axs[1].set_yscale("log"); axs[1].set_ylim(1e-3, None); axs[1].set_xlabel("t")
    axs[1].set_title("density at x = 2 (log scale)"); axs[1].legend(fontsize=7)
    _save(fig, out_dir, "part5_harsh.png")


def cmd_experiments():
    rows = run_experiments()
    Path("results").mkdir(exist_ok=True)
    for name, tex in summarize(rows).items():
        Path(f"results/table_{name}.tex").write_text(tex)
    fig_part5("figures")


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
