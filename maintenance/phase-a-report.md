# Nerva — Phase A Report: Empirical Baseline for the Python Backends and nerva-sympy

**Date:** 2026‑10‑31.
**Scope:** nerva-numpy, nerva-jax, nerva-tensorflow, nerva-torch and nerva-sympy.

**Commits tested:**

| Repo | Commit |
|---|---|
| nerva-numpy | `4724ce0` |
| nerva-jax | `97b1c79` |
| nerva-tensorflow | `92c7a83` |
| nerva-torch | `b2d9b75` |
| nerva-sympy | `981739f` |

**Method:**
- All work ran in `/tmp/phaseA`, a scratch area. Repos were copied there with `git archive`, and every environment was a throwaway uv venv.
- No repository files were changed, nothing was committed, and no project dependencies were changed.
- Linux x86_64 with 4 CPUs and no GPU.
- MNIST was prepared with the repos' own `prepare_data.py`.

**Environment limits:**
- `download.pytorch.org` and `www.cs.toronto.edu` are blocked in this session. As a result, the CPU-only torch index could not be tested and CIFAR‑10 could not be downloaded.
- Torch was therefore installed from PyPI's default wheel, which bundles CUDA.
- All GPU statements below come from package metadata only.

---

## 1. Compatibility matrix: latest releases

Each cell was installed into a fresh venv, then the tests were run with `pytest -m "not slow"` from the snapshot's `tests/` directory. The PyPI cells (nerva-*==1.0.0) used the same snapshot tests.

| Backend | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |
|---|---|---|---|---|
| numpy (PyPI and source) | ✅ 34 pass (numpy 2.4.6) | ✅ (numpy 2.5.4) | ✅ | ✅ |
| jax (PyPI and source) | ✅ 34 pass (jax 0.10.2) | ✅ (jax 0.12.0.1) | ✅ | ✅ |
| torch (PyPI and source) | ✅ 35 pass (torch 2.15.0) | ✅ | ✅ | ✅ |
| tensorflow (PyPI and source) | ✅ 34 pass (TF 2.21.0, keras 3.15.1) | ✅ | ✅ | ❌ **cannot install**: TF has no 3.14 wheel. With `--prerelease=allow`, TF 2.22.0rc0 installs and ✅ passes, but it pulls in `keras-nightly` |

- Every cell skips exactly one test, `test_one_layer.py`, because the golden `one_layer_cases/manifest.json` is not committed.
- PyPI 1.0.0 and the current source behave the same everywhere.
- **On Python 3.11, pip silently resolves to older framework releases:** numpy 2.4.6 and jax 0.10.2, because the latest numpy and jax require Python ≥3.12.

**Conclusion:**
- The code is compatible with Python 3.11–3.14 and the current framework releases.
- **TensorFlow is the only blocker for 3.14**, and that is a wheel-availability problem, not a code problem.

## 2. Declared lower bounds vs reality

**Resolver experiments.** Using `uv pip compile --resolution lowest-direct`:
- Without any restriction, the resolver happily produces combinations such as numpy 1.21 on Python 3.14. These have no wheels and would need a source build.
- Restricted to binary wheels, the lowest installable versions are:

| Python | numpy | torch | tensorflow | jax / jaxlib |
|---|---|---|---|---|
| 3.11 | 1.23.2 | 2.0.0 | 2.13.0 | 0.4.16 / 0.4.18 |
| 3.12 | 1.26.0 | 2.2.0 | 2.16.1 | 0.4.16 / 0.4.18 |
| 3.13 | 2.1.0 | 2.6.0 | 2.20.0 | 0.4.16 / 0.4.34 |
| 3.14 | 2.3.2 | 2.9.0 | ❌ | 0.4.16 / 0.7.1 |

**Installed and tested:**

| Combination | Result |
|---|---|
| numpy 1.23.2 + scikit-learn 1.1.3 (py3.11) | ✅ 34 pass |
| torch 2.0.0 + numpy 1.23.2 (py3.11) | ✅ 35 pass |
| tensorflow 2.13.0 (keras 2.13) + numpy 1.23.2 (py3.11) | ✅ 34 pass |
| jax 0.4.16 + jaxlib 0.4.18 (py3.11) | ❌ **broken**: `RuntimeError: jaxlib version 0.4.18 is newer than and incompatible with jax version 0.4.16` |
| jax 0.4.16 + jaxlib 0.4.34 (py3.13) | ❌ broken (same error) |
| jax 0.4.30 + jaxlib 0.4.30 (py3.11) | ✅ 34 pass |

**Findings:**
- The declared floors `numpy>=1.21`, `torch>=1.12`, `tensorflow>=2.10` and `jax>=0.4` are never what gets installed or tested.
- **Declaring `jaxlib` separately in nerva-jax lets the resolver choose an incompatible jax/jaxlib pair.** Recommendation: declare only `jax`.
- The library code itself is tolerant of old versions: torch 2.0, TF 2.13 with Keras 2, and numpy 1.23 all pass.

## 3. Install size and complexity

| Environment | venv size | Notes |
|---|---|---|
| nerva-numpy | 211 MB | about 150 MB of this is scikit-learn and scipy, used by a single example |
| nerva-jax | ~550 MB | |
| nerva-tensorflow | ~2.2 GB | |
| nerva-torch (PyPI default) | **5.5 GB** | torch 1.1 GB, **nvidia libs 3.0 GB, triton 1.0 GB**: 19 CUDA packages, all unused because nerva-torch never moves tensors to a GPU. `torchvision` (declared) is never imported |
| **nerva-sympy 1.0.0 from PyPI** | **7.8 GB, 77 packages, 15 nvidia packages**, ~6–7 min | Python 3.12 and 3.13 only. **Fails on 3.14** because of TensorFlow |
| nerva-sympy library alone (numpy + sympy, `--no-deps`) | **96 MB** | All library modules import fine without any framework |

**Conflicts.** I observed no resolver conflict between JAX, TensorFlow and PyTorch at today's versions. The all-framework environment resolves to protobuf 7.36.2, ml-dtypes 0.6.0 and numpy 2.5.4. The cost is size and the Python-version ceiling, not conflicts.

## 4. CPU/GPU findings

- **CPU-only torch:** the CPU index (`download.pytorch.org/whl/cpu`) could not be reached here. In a normal environment it should reduce the torch install from 5.5 GB to roughly 1 GB. To be confirmed in CI.
- **GPU behaviour:** unchanged from the Stage 2 analysis. JAX and TF will use a GPU automatically; torch code is CPU-only. I couldn't test this without a GPU.

## 5. nerva-sympy: tiered validation, verified

| Experiment | Result |
|---|---|
| Full suite, all frameworks (py3.13, seed 42) | ✅ 119 passed, 1 skipped (missing golden cases). **31 minutes** |
| Same suite, with "slow" tests included | Same 119 tests. **No test carries `@pytest.mark.slow`**, so the runner's `--include-long` option does nothing |
| Hyphenated `test-equations-*.py`, run explicitly | ✅ 5 passed in 0.13 s. Never collected by default |
| **Minimal environment** (numpy + sympy only), full test dir | ❌ **18 of 18 test modules fail at import**: 13 × `No module named 'jax'` (from `tests/utilities.py`), 4 × `nerva_jax`, 1 × `torch`. This includes about 13 purely symbolic modules |
| **Equation-check scripts in the minimal environment** (no frameworks, sibling source checkouts only) | ✅ **all 5 pass** (numpy, jax, tensorflow, torch, sympy): "Matching test equations were found for all layers" |

**This confirms the proposed tiers:**
- **Tier 1 (equation-text checks):** works today with zero frameworks installed.
- **Tier 0 (symbolic tests):** only needs the module-level framework imports in `tests/utilities.py` made lazy.

**Runtime.** The ~31-minute runtime comes from a few known slow symbolic tests. Marking them `slow` is deferred at the maintainer's request.

## 6. `test_frameworks.py`: root cause of the flakiness, and a fix

**Failure rate.** With the default absolute-error loss, the test **failed for seed 2**, which was 1 of 8 seeds and later 1 of 20. The failure was `DY mismatch at step 42: 0.0199 > 1e-5`.

**Diagnosis:**
- Y agrees across all backends to within 1e‑6.
- At one element where Y ≈ 0, torch computed Y = −5.7e‑8 while numpy, jax and TF computed about +1e‑7.
- The custom `AbsoluteErrorLossFunction` has gradient `sign(Y − T)`, so DY flipped from +0.01 to −0.01.
- So this is not a backend bug. A loss with a discontinuous gradient turns float32 rounding noise into an O(1) difference.

**Proposed replacement custom loss: pseudo-Huber.** With E = Y − T:
- loss: L = Σ δ²(√(1 + E²/δ²) − 1)
- gradient: DY = E ⊙ inv_sqrt(1 + E²/δ²)

Properties:
- It is smooth everywhere and abs-like for |E| ≫ δ, with the gradient bounded by δ.
- It uses only existing primitives (`square`, `inv_sqrt`, `hadamard`, `elements_sum`), which all five backends have.
- It still exercises the user-defined `LossFunction` path, and its gradient can be checked symbolically in SymPy.

**Test.** I patched only a scratch copy of `mlp_constructors.py` and ran 20 seeds per variant:

| Loss | Passed |
|---|---|
| absolute error (current) | 19/20 (seed 2 fails) |
| pseudo-Huber, δ = 0.1 | **20/20** |
| pseudo-Huber, δ = 1.0 | **20/20** |

**Margin.** For seed 2 with pseudo-Huber δ = 0.1, the largest DY deviation over 50 steps was **3.1e‑8**, against a 1e‑5 tolerance. The largest Y deviation was 4.7e‑6.

**Remaining recommendations for this test, independent of the loss:**
- Use a fixed default seed.
- Use a small synthetic dataset instead of MNIST.
- Move it to the optional all-framework tier (Tier 3).
- Rely on the golden fixtures for per-backend consistency.

## 7. Defects reproduced

**`tools/mlp.py`:**
- **Default `--layer-sizes`:** crashes with `ValueError: invalid literal for int() with base 10: '3072,128,64,10'`. The default is comma-separated, but the code splits on `;`.
- **Missing `--layers`:** gives `AttributeError: 'NoneType' object has no attribute 'split'`.

**`examples/train_mnist_cli.sh`:**
- **Without the package installed:** fails with `ModuleNotFoundError: No module named 'nerva_numpy'`, because `PYTHONPATH=../src` is never exported.
- **With the package installed:** runs correctly. One MNIST epoch took about 80 s for nerva-numpy, measured on a loaded machine.

**`test_one_layer.py`:** silently skipped in all 32 matrix cells and in nerva-sympy, because the golden cases are not committed.

## 8. Recommendations from Phase A (input for the Phase B backlog)

1. **Supported Python versions:** 3.11–3.14. TensorFlow joins 3.14 when 2.22 is final.
2. **CI matrix:** numpy, jax and torch on 3.11–3.14; TF on 3.11–3.13. Add one "minimum versions" job pinned to the tested floors: numpy 1.23, torch 2.0, TF 2.13, jax = jaxlib 0.4.30.
3. **nerva-jax:** remove the separate `jaxlib` requirement.
4. **Optional dependencies:** remove `torchvision`; move `scikit-learn` to an `[examples]` extra; install torch in CI from the CPU index.
5. **nerva-sympy:**
   - Core dependencies become `numpy` and `sympy`; per-backend extras plus `[all]`.
   - Make the framework imports in `tests/utilities.py` lazy.
   - Rename the equation tests to `test_equations_*.py`.
   - Commit the golden one-layer cases.
   - Deferred: marking the known slow tests.
6. **`test_frameworks.py`:** switch the custom loss to pseudo-Huber, fix the default seed, use synthetic data, and move it to the nightly tier.
7. **Fix the CLI and example-script defects** in §7.

## Reproducing
- **Scripts:** `/tmp/phaseA/scripts/`. `run_cell.sh` runs one matrix cell; `sympy_exp.sh` runs the nerva-sympy experiments.
- **Logs:** `/tmp/phaseA/logs/`.
- **Seed-sweep harness:** `/tmp/phaseA/diag/`.
- These scratch paths are temporary and will be lost when the session container is reclaimed.
