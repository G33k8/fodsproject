# Inverse Problem MDN (MAC 312 Project 8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and compare an MSE network, a single-Gaussian network and a mixture density network (MDN) that predict t from x = t² + noise, all graded against an exact quadrature reference p(t | x), and write the Part 1–5 report.

**Architecture:** Five flat Python modules at the repo root (`data.py`, `reference.py`, `models.py`, `train.py`, `evaluate.py`), as the PRD names them. All three models share one MLP body; the single-Gaussian network is `MDN(M=1)`. `evaluate.py` holds the metrics, every figure, and the experiment-matrix runner, behind a two-command CLI. The report is LaTeX in `report/`, and its appendix pulls in the five files with `\lstinputlisting` so the listings can't drift from the code.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch (CPU), Matplotlib, pytest.

**Spec:** `PRD Project 8 – Inverse Problems with Multiple Valid Solutions.md` (repo root). Read it alongside this plan.

## Global Constraints

- Defaults: `a = 2`, `σ = 0.2`, train/val/test = 5,000 / 1,000 / 2,000, reference grid = 2,001 points on [−a, a], body = 1 → 64 → 64 tanh, `M = 3`, variance = `softplus(raw) + 1e-4`, Adam lr `1e-3`, batches of 256, up to 3,000 epochs with early stopping on validation NLL.
- Use PyTorch autograd. No hand-written backprop.
- MDN loss: `log_softmax` for log π, closed-form Gaussian log density per component, then `torch.logsumexp`. Never take the log of an already-summed density.
- Single-Gaussian baseline = `MDN(M=1)`, the same class and loss.
- Never tune on the test set. Early stopping and model selection use validation only.
- Every figure and table must be reproducible from a fixed seed. Data seed is fixed at 0 for all experiments; the "seed" in the experiment matrix controls model init and batch order only.
- Normalize with the trapezoid rule. Use `scipy.integrate.trapezoid`, which is the same rule as `np.trapz`. `np.trapz` is deprecated or removed in NumPy 2.x. Say this in one line in the report.
- Lead the evaluation with log-likelihood and density plots. Test MSE is reported only to show it misleads.
- Compare the forward residuals x − t² against N(0, σ²), not against zero.
- Figures go in `figures/` (PNG, dpi 150). Results go in `results/`.
- Deadline Nov 21, 2026. Task-to-day mapping: Tasks 1–2 Nov 18, Tasks 3–4 Nov 19, Tasks 5–6 Nov 20, Tasks 7–8 Nov 21.

## Review Focus

1. **x far outside the training range** (x = 10 ≫ a², x = −1). A naive `exp(-(x−t²)²/2σ²)` underflows to all zeros, so normalization becomes 0/0 and returns NaN. Expected: a finite density that integrates to 1. The fix is to normalize in log space (subtract the max). Pinned in Task 2 `test_far_x_is_finite`.
2. **Shape broadcasting in the losses.** MSE prediction `(N,1)` minus target `(N,)` silently broadcasts to `(N,N)` and gives a plausible-looking wrong loss. Expected: the loss equals the per-sample mean. Pinned in Task 4 `test_mse_loss_no_broadcast`.
3. **Variance floor = 0 (a Part 5 setting).** The loss can become NaN or −∞ mid-training. Expected: the run stops cleanly with `diverged=True` and the experiment matrix keeps going. Pinned in Task 4 `test_nan_loss_sets_diverged` and Task 7 `test_run_one_survives_divergence`.
4. **Very rare branch (P(t<0) = 0.05, about 250 negative training points).** The reference must give the correct asymmetric branch probability, and the MDN may lose the branch. Expected: the reference is correct, and the experiment records a `rare_branch_recovered` flag rather than crashing or hiding it. Pinned in Task 2 `test_branch_prob_asymmetric` and Task 7 metrics.
5. **Leakage and reproducibility.** The test split must differ from the train/val splits, and the same seed must give bit-identical trained weights. Pinned in Task 1 `test_splits_differ` and Task 4 `test_same_seed_same_weights`.

---

### Task 1: Environment and data generation

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `sample_t(n: int, a: float, p_neg: float, rng: np.random.Generator) -> np.ndarray`: shape `(n,)`. Each draw is negative with probability `p_neg`, and uniform on its half: `Unif[−a, 0)` if negative, `Unif[0, a]` if not. With `p_neg = 0.5` this is exactly `Unif[−a, a]`.
  - `make_dataset(n: int, a: float = 2.0, sigma: float = 0.2, p_neg: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]`: returns `(x, t)`, both float64 with shape `(n,)`, where `x = t² + N(0, σ²)`.
  - `make_splits(a=2.0, sigma=0.2, p_neg=0.5, seed=0) -> dict[str, tuple[np.ndarray, np.ndarray]]`: keys `"train"`, `"val"`, `"test"`, with sizes 5000 / 1000 / 2000 and data seeds `seed`, `seed + 1000`, `seed + 2000`.

- [ ] **Step 1: Set up the environment**

`requirements.txt`: `numpy`, `scipy`, `matplotlib`, `torch`, `pytest`.
`.gitignore`: `.venv/`, `__pycache__/`, `.pytest_cache/`, `report/*.aux`, `report/*.log`, `report/*.out`.
`pytest.ini`: register the marker `slow: trains real models (minutes)`. Set `addopts = -m "not slow"` so the default run stays fast.

Run:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib pytest
```
Expected: `python -c "import torch, numpy, scipy, matplotlib"` exits 0. torch is not installed system-wide, so every later command assumes `.venv` is active.

- [ ] **Step 2: Write the failing tests** in `tests/test_data.py`

```python
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
```

- [ ] **Step 3: Run the tests and confirm they fail.** Run `pytest tests/test_data.py -v`. Expected: ImportError.
- [ ] **Step 4: Implement `data.py`** with the three functions above, using `np.random.default_rng(seed)`.
- [ ] **Step 5: Run the tests and confirm they pass.** Run `pytest tests/test_data.py -v`. Expected: 6 passed.
- [ ] **Step 6: Commit.** `git add requirements.txt .gitignore pytest.ini data.py tests/test_data.py && git commit -m "feat: data generation with symmetric/asymmetric prior"`

---

### Task 2: Exact reference p(t | x)

**Files:**
- Create: `reference.py`
- Test: `tests/test_reference.py`

**Interfaces:**
- Produces (`x` may be a scalar or a 1-D array; output rows follow `np.atleast_1d(x)`):
  - `t_grid(a: float = 2.0, n_grid: int = 2001) -> np.ndarray`: `np.linspace(-a, a, n_grid)`.
  - `posterior(x, a=2.0, sigma=0.2, p_neg=0.5, n_grid=2001) -> tuple[np.ndarray, np.ndarray]`: returns `(t, density)` with `density.shape == (len(x), n_grid)`. log density = log prior(t) − (x − t²)²/(2σ²), where prior(t) = `p_neg/a` for t < 0 and `(1−p_neg)/a` for t ≥ 0. Subtract the per-row max, exponentiate, then divide by `scipy.integrate.trapezoid(·, t)`.
  - `branch_prob(x, a=2.0, sigma=0.2, p_neg=0.5) -> np.ndarray`: P(t > 0 | x), the trapezoid integral of the density over `t >= 0`.
  - `posterior_mean(x, a=2.0, sigma=0.2, p_neg=0.5) -> np.ndarray`: E[t | x], the target an MSE network converges to.
  - `log_density_at(x: np.ndarray, t: np.ndarray, a=2.0, sigma=0.2, p_neg=0.5) -> np.ndarray`: the exact log p(tᵢ | xᵢ) per pair, using the unnormalized log density at tᵢ minus the log normalizer for xᵢ (from the grid). Returns `-inf` for |t| > a. This is the "exact" row of the log-likelihood table.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_reference.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `reference.py`** per the Interfaces block. Use broadcasting `(len(x), 1)` against `(n_grid,)`. No loops.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `pytest tests/test_reference.py -v`. Expected: 8 passed.
- [ ] **Step 5: Commit.** `git add reference.py tests/test_reference.py && git commit -m "feat: exact posterior reference by quadrature"`

---

### Task 3: Models and MDN loss

**Files:**
- Create: `models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `class Body(nn.Module)`: `Linear(1,64) → tanh → Linear(64,64) → tanh`. `forward(x: Tensor (N,1)) -> Tensor (N,64)`.
  - `class MSENet(nn.Module)`: `Body` + `Linear(64,1)`. `forward(x (N,1)) -> Tensor (N,)`. Squeeze the last dim.
  - `class MDN(nn.Module)`: `__init__(self, M: int = 3, var_floor: float = 1e-4, a: float = 2.0)`. `Body` + `Linear(64, 3M)` split into logits, mu, raw_var. Initialize the mu-slice bias to `torch.linspace(-a, a, M)` (for M = 1 that gives 0), which is the PRD's mitigation against collapse. `forward(x (N,1)) -> tuple[log_pi, mu, var]`, each `(N, M)`, with `log_pi = log_softmax(logits)` and `var = softplus(raw_var) + var_floor`.
  - `mdn_log_prob(log_pi, mu, var, t: Tensor (N,)) -> Tensor (N,)`: `logsumexp(log_pi − 0.5·(log(2π·var) + (t[:,None]−mu)²/var), dim=1)`.
  - `mdn_nll(log_pi, mu, var, t) -> Tensor` (scalar): `-mdn_log_prob(...).mean()`. This is the PRD sum divided by N, so its value doesn't depend on batch size. Say so in the report.
  - `mdn_mean(log_pi, mu) -> Tensor (N,)`: Σ πⱼ μⱼ, used for the MDN's test MSE.
  - `mdn_branch_prob(log_pi, mu, var) -> Tensor (N,)`: Σ πⱼ · Φ(μⱼ / √vⱼ), the closed-form P(t > 0).
  - `mdn_sample(log_pi, mu, var, generator: torch.Generator) -> Tensor (N,)`: one draw per row. Pick a component with `torch.multinomial(log_pi.exp(), 1, generator=...)`, then draw from its Gaussian.
  - `to_tensor(x: np.ndarray) -> Tensor`: float32 `(N,1)` for inputs. Targets use `torch.as_tensor(t, dtype=torch.float32)` with shape `(N,)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_mdn_shapes_and_constraints():
    log_pi, mu, var = MDN(M=3)(torch.linspace(-0.5, 4.5, 7)[:, None])
    assert log_pi.shape == mu.shape == var.shape == (7, 3)
    torch.testing.assert_close(log_pi.exp().sum(1), torch.ones(7))
    assert (var >= 1e-4).all()

def test_mean_bias_spread():
    torch.testing.assert_close(MDN(M=3, a=2.0).head.bias[3:6], torch.tensor([-2.0, 0.0, 2.0]))
    # assumes head output order [logits(M), mu(M), raw_var(M)] and attribute name `head`

def test_log_prob_matches_bruteforce():
    log_pi = torch.log(torch.tensor([[0.3, 0.7]]))
    mu, var, t = torch.tensor([[-1.0, 1.0]]), torch.tensor([[0.04, 0.09]]), torch.tensor([0.8])
    expected = np.log(0.3 * norm.pdf(0.8, -1, 0.2) + 0.7 * norm.pdf(0.8, 1, 0.3))
    assert abs(mdn_log_prob(log_pi, mu, var, t).item() - expected) < 1e-5

def test_log_prob_stable_far_from_all_components():
    log_pi = torch.log(torch.tensor([[0.5, 0.5]]))
    mu, var = torch.tensor([[0.0, 1.0]]), torch.tensor([[1e-4, 1e-4]])
    lp = mdn_log_prob(log_pi, mu, var, torch.tensor([100.0]))
    assert torch.isfinite(lp).all()   # naive log(sum(pdf)) gives -inf here

def test_branch_prob_matches_monte_carlo():
    log_pi = torch.log(torch.tensor([[0.2, 0.8]]))
    mu, var = torch.tensor([[-1.0, 0.3]]), torch.tensor([[0.25, 0.5]])
    g = torch.Generator().manual_seed(0)
    s = torch.stack([mdn_sample(log_pi, mu, var, g) for _ in range(20000)])
    assert abs((s > 0).float().mean() - mdn_branch_prob(log_pi, mu, var)[0]) < 0.01

def test_mse_net_output_is_1d():
    assert MSENet()(torch.zeros(5, 1)).shape == (5,)
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_models.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `models.py`** per the Interfaces block. Name the output layer `head` in both nets.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `pytest tests/test_models.py -v`. Expected: 6 passed.
- [ ] **Step 5: Commit.** `git add models.py tests/test_models.py && git commit -m "feat: shared MLP body, MSE net, stable MDN"`

---

### Task 4: Training loop

**Files:**
- Create: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `MSENet`, `MDN`, `mdn_nll`, `to_tensor` (Task 3), and `make_splits` (Task 1).
- Produces:
  - `set_seed(seed: int) -> None`: seeds torch, numpy and `random`.
  - `mse_loss(model: MSENet, x: Tensor (N,1), t: Tensor (N,)) -> Tensor`: the mean squared error.
  - `nll_loss(model: MDN, x, t) -> Tensor`: `mdn_nll(*model(x), t)`.
  - `train(model, loss_fn, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray], epochs: int = 3000, lr: float = 1e-3, batch_size: int = 256, patience: int = 200, clip: float = 5.0, seed: int = 0) -> dict`. Returns `{"train": list[float], "val": list[float], "best_epoch": int, "best_val": float, "diverged": bool}`. It mutates `model` in place to the best-validation weights. Shuffle with a `torch.Generator` seeded from `seed`. Clip with `clip_grad_norm_(…, clip)`. Stop after `patience` epochs without validation improvement. If any batch loss or validation loss is non-finite, set `diverged=True`, stop, and restore the best weights. Do not raise.
  - `fit_model(kind: str, splits: dict, seed: int = 0, M: int = 3, var_floor: float = 1e-4, a: float = 2.0, **train_kw) -> tuple[nn.Module, dict]`. `kind` is one of `"mse"`, `"gauss"` (= `MDN(M=1)`) or `"mdn"`. It calls `set_seed(seed)` before building the model, then calls `train` and returns `(model, history)`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_train.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `train.py`** per the Interfaces block. Compute the validation loss under `torch.no_grad()` on the full validation set each epoch. Save the best weights with `copy.deepcopy(model.state_dict())`.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `pytest tests/test_train.py -v`. Expected: 4 passed.
- [ ] **Step 5: Time one full run.** Run `python -c "import time,data,train;s=data.make_splits();t0=time.time();_,h=train.fit_model('mdn',s);print(time.time()-t0,h['best_epoch'],h['best_val'],h['diverged'])"`. Expected: `diverged False`, a finite `best_val` of about 0.0 or below, and the time printed. Write the seconds-per-run figure in the commit message, because Task 7 runs about 80 of these.
- [ ] **Step 6: Commit.** `git add train.py tests/test_train.py && git commit -m "feat: training loop with early stopping and divergence guard"`

---

### Task 5: Evaluation metrics and success-criteria tests

**Files:**
- Create: `evaluate.py` (the metrics section only)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces (all run under `torch.no_grad()` and return NumPy):
  - `mean_loglik(model: MDN, x: np.ndarray, t: np.ndarray) -> float`: the mean of `mdn_log_prob`.
  - `predict_mean(model, x) -> np.ndarray`: `MSENet` output, or `mdn_mean` for an MDN.
  - `test_mse(model, x, t) -> float`.
  - `model_density(model: MDN, x: float, t: np.ndarray) -> np.ndarray`: the mixture pdf on the grid `t`.
  - `model_branch_prob(model: MDN, x: np.ndarray) -> np.ndarray`.
  - `forward_residuals(model: MDN, x: np.ndarray, seed: int = 0) -> np.ndarray`: draw one `mdn_sample` per test x and return `x − t_s²`. Marginally over the test set this is distributed as N(0, σ²) when the model is correct. Put that sentence in the docstring.
  - `BRANCH_X = np.linspace(0.5, 3.5, 61)`: the "x clearly above 0" grid that branch-probability error is measured on.
  - `branch_error(model, a, sigma, p_neg) -> float`: `max |model_branch_prob − reference.branch_prob|` over `BRANCH_X`.

- [ ] **Step 1: Write the failing tests.** These are the PRD success criteria. Give the module a fixture, `scope="module"`, that trains `mse`, `gauss` and `mdn` once on `make_splits(seed=0)` with seed 0.

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_evaluate.py -v -m "slow or not slow"`. Expected: ImportError.
- [ ] **Step 3: Implement the metrics section of `evaluate.py`** per the Interfaces block.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `pytest tests/test_evaluate.py -v -m "slow or not slow"`. Expected: 5 passed. If `test_mdn_branch_prob_within_005` fails, debug with superpowers:systematic-debugging. Check component collapse (print π, μ at x = 2) before you touch thresholds. Do not loosen the PRD's 0.05.
- [ ] **Step 5: Commit.** `git add evaluate.py tests/test_evaluate.py && git commit -m "feat: evaluation metrics and success-criteria tests"`

---

### Task 6: Figures for Parts 1, 2 and 4

**Files:**
- Modify: `evaluate.py` (add the plotting section and a CLI)
- Test: `tests/test_evaluate.py` (add a smoke test)

**Interfaces:**
- Consumes: the Task 5 metrics and the Task 4 `fit_model`.
- Produces:
  - `FIG_X = [-0.2, 0.1, 1.0, 3.0, 4.5]`.
  - `fig_reference(out_dir: Path) -> Path`: writes `figures/part1_reference.png`, one panel per x in `FIG_X`, with ±√x marked where x > 0.
  - `fig_mse(models: dict, splits, out_dir) -> Path`: writes `part2_mse.png`. Training scatter (t on the y-axis, x on the x-axis), the MSE predictions, `reference.posterior_mean`, and the ±√x curves.
  - `fig_overlays(models, out_dir) -> Path`: writes `part4_overlays.png`. For each x in `FIG_X`, plot the reference, the single Gaussian and the MDN.
  - `fig_branch_prob(models, out_dir) -> Path`: writes `part4_branch_prob.png`. Reference vs MDN on `np.linspace(-0.5, 4.5, 201)`, with a horizontal line at 0.5 and `BRANCH_X` shaded.
  - `fig_residuals(models, splits, sigma, out_dir) -> Path`: writes `part4_residuals.png`. Density histograms for the single Gaussian and the MDN, with the N(0, σ²) pdf overlaid.
  - `table_baseline(models, splits) -> str`: a LaTeX `tabular` with rows Exact reference / MSE / Single Gaussian / MDN and columns test mean log-likelihood and test MSE. The reference uses `reference.log_density_at` and `posterior_mean`. MSE has no log-likelihood, so print "—" there.
  - CLI: `python evaluate.py figures` trains the three seed-0 models, writes all figures above, and writes `results/table_baseline.tex`.

- [ ] **Step 1: Write the failing smoke test.** `test_fig_reference_writes_file(tmp_path)` calls `fig_reference(tmp_path)` and asserts the returned path exists and is larger than 10 kB. Use the `matplotlib.use("Agg")` backend at the top of `evaluate.py`.
- [ ] **Step 2: Run the test and confirm it fails.** Run `pytest tests/test_evaluate.py::test_fig_reference_writes_file -v`. Expected: AttributeError.
- [ ] **Step 3: Implement the plotting functions and the `argparse` CLI** with the subcommands `figures` and `experiments` (the second one is filled in by Task 7).
- [ ] **Step 4: Run and inspect.** Run `pytest tests/test_evaluate.py -v && python evaluate.py figures`. Expected: the tests pass and 5 PNGs plus `results/table_baseline.tex` exist. Open each PNG with the Read tool and check it: two peaks at ±1 for x = 1 in the reference and the MDN; one wide bell for the single Gaussian; MSE predictions that hug 0.
- [ ] **Step 5: Commit.** `git add evaluate.py tests/test_evaluate.py figures results && git commit -m "feat: Part 1/2/4 figures and baseline table"`

---

### Task 7: Experiment matrix and Part 5 figures

**Files:**
- Modify: `evaluate.py` (add the experiments section)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Tasks 4–6.
- Produces:
  - `SEEDS = [0, 1, 2, 3, 4]`
  - `EXPERIMENTS: dict[str, list[dict]]`: the PRD matrix, where every config is `{kind, sigma, p_neg, M, var_floor}` with defaults `σ=0.2, p_neg=0.5, M=3, var_floor=1e-4`:
    - `baseline`: kind ∈ {mse, gauss, mdn}
    - `noise`: σ ∈ {0.05, 0.2, 0.5}
    - `prior`: p_neg ∈ {0.5, 0.2, 0.05}
    - `components`: M ∈ {1, 2, 3, 5}
    - `floor`: var_floor ∈ {0.0, 1e-4, 1e-2}
    - Initialization = the 5-seed spread that every row already carries. It is not a separate experiment.
  - `run_one(cfg: dict, seed: int) -> dict`: builds `make_splits(sigma=cfg["sigma"], p_neg=cfg["p_neg"], seed=0)`, calls `fit_model`, and returns `cfg | {seed, test_loglik, test_mse, branch_error, residual_std, rare_branch_recovered, best_epoch, diverged}`. Fields that don't apply (log-likelihood for mse) are `None`, as are metrics that are non-finite. `rare_branch_recovered` = model P(t<0 | x=2) > 0.5 × reference P(t<0 | x=2).
  - `run_experiments(path: Path = Path("results/experiments.json")) -> list[dict]`: resumable. It loads existing rows, skips (cfg, seed) pairs already present, and appends and saves after each run.
  - `summarize(rows) -> dict[str, str]`: one LaTeX table per experiment, mean ± std over seeds, plus a "diverged k/5" column. Written to `results/table_<name>.tex`.
  - `fig_part5(rows_or_models, out_dir)`: writes `part5_prior.png` (branch-probability curves and density at x = 1 for each p_neg), `part5_components.png` (density at x ∈ {1, 3} for each M), `part5_floor.png` (validation-NLL curves for each floor, with divergence marked, and the density at x = 1), and `part5_noise.png` (MDN vs reference at x = 1 for each σ). Each one retrains only the seed-0 models it needs.
  - CLI: `python evaluate.py experiments` runs the matrix, the tables and the Part 5 figures.

- [ ] **Step 1: Write the failing tests**

```python
def test_experiment_matrix_shape():
    assert {k: len(v) for k, v in EXPERIMENTS.items()} == {"baseline": 3, "noise": 3, "prior": 3, "components": 4, "floor": 3}

def test_run_one_survives_divergence(monkeypatch):
    monkeypatch.setattr("train.train", lambda *a, **k: {"train": [], "val": [float("nan")], "best_epoch": 0, "best_val": float("nan"), "diverged": True})
    row = run_one({"kind": "mdn", "sigma": 0.2, "p_neg": 0.5, "M": 3, "var_floor": 0.0}, seed=0)
    assert row["diverged"] is True
    json.dumps(row)  # serializable, NaN mapped to None

def test_run_experiments_resumes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate, "run_one", lambda cfg, seed: calls.append(1) or {**cfg, "seed": seed})
    p = tmp_path / "e.json"; run_experiments(p); n = len(calls); run_experiments(p)
    assert len(calls) == n  # second call runs nothing
```

(If `fit_model` imports `train` by name, patch `train.train` where `fit_model` looks it up. Adjust the patch target, not the assertion.)

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_evaluate.py -v`. Expected: the 3 new tests fail.
- [ ] **Step 3: Implement** the Interfaces block.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `pytest -v`. Expected: all fast tests pass.
- [ ] **Step 5: Run the matrix in the background.** Run `python evaluate.py experiments`. That is 80 runs. Use the seconds-per-run figure from Task 4. If the estimate is over 90 minutes, pass `batch_size=1000` to every matrix run (not to the Task 6 baseline figures) and state that in the report. Expected: `results/experiments.json` with 80 rows, 5 `table_*.tex` files, and 4 `part5_*.png`. Look at the PNGs.
- [ ] **Step 6: Commit.** `git add evaluate.py tests/test_evaluate.py results figures && git commit -m "feat: experiment matrix, Part 5 stress tests"`

---

### Task 8: Report and LaTeX appendix

**Files:**
- Create: `report/report.tex`, `report/appendix.tex`

**Interfaces:**
- Consumes: every file in `figures/` and `results/table_*.tex` (via `\input`).

- [ ] **Step 1: Write `report/appendix.tex`.** One `\section*` per file, each with `\lstinputlisting[language=Python]{../<file>.py}` for `data`, `reference`, `models`, `train` and `evaluate`. This settles the PRD's open question, "full code in a LaTeX appendix".
- [ ] **Step 2: Write `report/report.tex`** (`article`, packages `amsmath`, `graphicx`, `listings`, `booktabs`). Follow PRD Parts 1–5 in order:
  - **Part 1:** derive p(t | x) by Bayes with the flat (and asymmetric) prior, and describe quadrature normalization. Write up the four regions: x ≤ 0 (one blob at 0); x clearly > 0 (two peaks at ±√x of width ≈ σ/(2√x)); x near a² (peaks pushed against the ±a boundary and truncated); x ≫ a² (mass piled at ±a, and the model is extrapolating). Include `part1_reference.png`.
  - **Part 2:** derive that the MSE minimizer is E[t | x], which is 0 under the symmetric prior. Include `part2_mse.png`.
  - **Part 3:** derive the MDN NLL from the PRD formula, and explain log_softmax, softplus + floor and logsumexp.
  - **Part 4:** include `table_baseline.tex` and `table_baseline` from the matrix (mean ± std), the overlays, the branch-probability figure and the residual figures. Say why residuals are compared to N(0, σ²).
  - **Part 5:** include the noise, prior, components and floor tables and figures. Report rare-branch recovery honestly, give the number of diverged runs for floor = 0, and state the written argument: MSE scores only E[t | x], so two models with the same mean but one or two peaks get the same MSE. Cite the test-MSE column as evidence.
  - Quote numbers only from the `results/` files. Don't invent any.
- [ ] **Step 3: Compile.** Run `cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex`. Expected: `report.pdf` is written with no undefined references. If `pdflatex` is not installed, say so and leave the `.tex` for the user to compile (for example on Overleaf). Do not install TeX.
- [ ] **Step 4: Run the final check.** Run `pytest -v -m "slow or not slow"`. Expected: all tests pass.
- [ ] **Step 5: Commit.** `git add report && git commit -m "docs: Project 8 report and code appendix"`
