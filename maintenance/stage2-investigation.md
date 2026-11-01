# Nerva — Stage 2 Investigation: Python Backends & nerva-sympy

## Context
Revised priorities from the maintainer:
- **In scope:** the four Python backends (nerva-numpy, nerva-jax, nerva-tensorflow, nerva-torch), plus nerva-sympy.
- **C++:** MKL stays the preferred backend. No investigation of, or work on, an Eigen-only or other alternative backend.

This document gives the findings for the 7 investigation questions plus nerva-sympy, and a proposed execution plan.

**Method so far:** read-only. Static analysis of all five repos (local clones), plus live PyPI metadata queried on 2026‑10‑31. Nothing was installed, modified, committed or upgraded. The installation claims below are therefore *predictions from metadata*. They still need to be verified empirically in Phase B.

---

## 1. Compatibility with current Python versions

**Current state of the code.** Every package declares `requires-python >=3.10` with no Python-version classifiers.
- The source uses no syntax newer than 3.10 and no removed NumPy‑2 aliases (`np.float_` etc.), Keras, or removed JAX APIs.
- Some legacy torch constructors (`torch.Tensor(...)`, `torch.LongTensor(...)`) still work but are discouraged. They appear in `nerva_torch/utilities.py:151`, `activation_functions.py:181`, `layers.py:156` and `examples/train_synthetic.py`.

**Python status today.** 3.10 has just reached end of life (Oct 2026). 3.11 is security-fixes only. 3.12, 3.13 and 3.14 are maintained, and 3.15 is newly released.

**Assessment.**
- The code itself is very probably compatible with 3.10–3.15.
- The real constraint is which frameworks publish wheels for which Python versions (see §2).
- `>=3.10` is now an EOL floor and should move up.

## 2. Current stable framework versions and constraints (from PyPI)

| Package | Latest | Python wheels | Nerva's declared floor |
|---|---|---|---|
| numpy | 2.5.4 (2026‑10) | 3.12–3.15 | `>=1.21` (no wheels for 3.11+) |
| jax / jaxlib | 0.12.0.1 (2026‑10) | 3.12–3.15 | `>=0.4` (2022) |
| torch | 2.15.0 (2026‑10‑28) | 3.11–3.15 | `>=1.12` (no wheels for 3.11+) |
| tensorflow | 2.21.0 (2026‑03) | **3.10–3.13 only**. 2.22.0rc0 (2026‑09) adds 3.14 | `>=2.10` |
| scikit-learn | 1.9.1 | 3.11–3.15 | `>=1.1` |
| sympy | 1.14.0 (pure Python) | ≥3.9 | `>=1.9` |

**Conclusions:**
- **TensorFlow is the bottleneck.** It has no 3.14 wheel until 2.22 is final, and no 3.15 wheel at all.
- **The declared lower bounds are fiction.** They have never been tested, and several of them cannot even be installed on the Python versions they claim to support.
- **Proposed support window:**
  - Python 3.11–3.14 for all four backends. 3.10 is dropped because it is EOL.
  - 3.15 for numpy, jax and torch now; TensorFlow joins once Google ships wheels.
  - This also implies new floors, e.g. numpy ≥1.26, torch ≥2.1–2.2, jax ≥0.4.30, tensorflow ≥2.16. These would be validated by a "minimum versions" CI job, not guessed.

## 3. Dependency conflicts and installation complexity

**Per-backend packages (each installed alone):**
- **`torchvision` is declared in nerva-torch but never imported anywhere.** It is pure overhead, and its version must match torch exactly.
- **`scikit-learn` is a hard runtime dependency of all four backends**, but it is used only in `examples/train_synthetic.py`.
- **`jaxlib` is declared alongside `jax`.** That is redundant: `jax` pins its own matching jaxlib, and a separate floor can only cause confusion.
- **`pytest` is used by the test runner but not declared anywhere.** There is no test extra.
- **`requirements.txt` duplicates `pyproject.toml`** without versions, so the two can drift.

**nerva-sympy is the main conflict and complexity hotspot.** Its *runtime* dependencies are all four backends, which pulls in JAX, TensorFlow and PyTorch together.
- That is several GB of downloads, because torch from PyPI on Linux includes CUDA libraries.
- The environment is limited to Python ≤3.13, the intersection with TensorFlow.
- Each framework pins its own versions of shared libraries: `ml-dtypes`, `protobuf` (TF requires `<8,>=6.31`), the `nvidia-*-cu12` libs (TF `[and-cuda]`), and CUDA 12 vs 13 (jax `cuda13`, torch). Pip would have to satisfy all of these at once.
- The library code itself does not need any of them (see §8).

**Docs CI:** `nerva-torch/.github/workflows/deploy-sphinx.yml` runs `pip install torch` even though `doc/sphinx/conf.py` already mocks `torch`. That is a wasted multi-GB download on every docs build.

## 4. CPU and GPU installation options

| Backend | CPU install | GPU install | What Nerva does with a GPU today |
|---|---|---|---|
| numpy | `pip install nerva-numpy` | n/a (CuPy would be a new backend; out of scope) | CPU only |
| jax | default wheel is CPU (`jax[cpu]`) | `jax[cuda12]` / `jax[cuda13]` (Linux), `jax[tpu]`, ROCm/oneAPI plugins | **JAX places arrays on the GPU automatically.** `disable_gpu()` exists in utilities but is never called by the library |
| tensorflow | `tensorflow-cpu`, or `tensorflow` (CPU on macOS/Windows) | `tensorflow[and-cuda]` (Linux/WSL2, CUDA 12). No native Windows GPU support since TF 2.11 | **TF also uses the GPU automatically** |
| torch | `pip install torch --index-url https://download.pytorch.org/whl/cpu` (the PyPI default on Linux bundles CUDA) | default PyPI wheel, or `--index-url …/whl/cu12x` | **Never uses the GPU.** There is no `device` handling anywhere, so all tensors stay on the CPU |

**Implications:**
- The "GPU not available" limitation in the README is inaccurate for the Python backends. JAX and TF already run on a GPU if one is present; torch would need a small device-handling layer.
- **GPU numerics differ.** TF enables TF32 matmuls on Ampere+ GPUs by default, and JAX's default GPU matmul precision is also reduced. Consistency tests must force full precision (`jax.default_matmul_precision("highest")`, `tf.config.experimental.enable_tensor_float_32_execution(False)`).
- **Packaging can't fully express GPU choices.** Torch's CPU/CUDA choice is made by index URL and cannot go in a pyproject extra. JAX and TF *can* be expressed as extras (e.g. `nerva-jax[cuda]` → `jax[cuda13]`, `nerva-tensorflow[and-cuda]`). The README should give a per-platform install table.

## 5. Packaging improvements and optional dependencies

These are proposals for later approval; none are made yet.
- **Core dependencies:** the framework and numpy only. Remove `torchvision` and the separate `jaxlib`. Move `scikit-learn` to an `[examples]` extra.
- **Extras:**
  - `[test]` (pytest; nerva-sympy for the validation tests)
  - `[examples]` (scikit-learn, sympy)
  - `[dev]` = test + docs + lint
  - `nerva-jax[cuda]` and `nerva-tensorflow[and-cuda]` as thin pass-throughs
- **Metadata:**
  - Raise the setuptools floor and use an SPDX `license = "BSL-1.0"` with `license-files`.
  - Add Python-version classifiers.
  - Add the missing `Documentation` URL for TensorFlow.
- **CLI:** `tools/mlp.py` is not packaged. Expose it as a `[project.scripts]` entry point (e.g. `nerva-torch-mlp`).
- **Drop or regenerate `requirements.txt`** (e.g. `-e .[test]`).
- **Release automation:** tag → build → PyPI Trusted Publishing. This replaces nerva-sympy's `scripts/make_distribution.sh` and `upload_distribution.sh`.

## 6. Cross-backend numerical consistency

**Existing machinery:**
- `nerva-sympy/tests/test_frameworks.py` builds the same MLP in all four backends with synchronized weights and compares the intermediates over 50 SGD steps (atol 6e‑5). The design is good. Its problems:
  - It needs all four frameworks in one environment.
  - It needs `data/mnist-flattened.npz`, which is not committed.
  - It picks a **random seed on every run** unless `NERVA_TEST_SEED` is set, so CI failures would be non-reproducible.
- `tests/test_one_layer.py` (nerva-sympy and all backends) compares against golden cases in `tests/one_layer_cases/`. **The cases are not committed**: only a README is there, so the test **silently skips** on a fresh clone. The README also points at a renamed file, `test_one_layer_forward_backward.py`.
- The backends are deterministic once weights are loaded: `DataLoader` does not shuffle, and `--load-weights` / `load_weights_and_bias` exist. Weight initialisation uses each framework's own RNG, so inits can never match by seed alone. Shared weights are the only correct basis for comparison.
- All four backends use float32 by design. JAX initialisers build float64 NumPy arrays and rely on `jnp.asarray` to downcast them to float32.

**Proposed approach: golden-reference fixtures.**
1. Generate small reference cases once, in float64. Use SymPy-evaluated values for tiny shapes and nerva-numpy in float64 for larger ones. Each case records:
   - inputs X and T, and the initial W and b
   - Y and DY
   - the gradients DW, Db and DX
   - the parameters after one optimiser step
   Cover each layer type, activation, loss and optimiser.
2. Commit the fixtures, at a few hundred KB, to nerva-sympy.
3. Each backend's CI compares against the fixtures with float32 tolerances, **with only its own framework installed**. Consistency is transitive through the reference.
4. Keep the all-framework `test_frameworks.py` as an optional nightly job. It gets a fixed seed and a small synthetic dataset instead of MNIST.
5. Document the tolerance policy (rtol/atol per op class) and the GPU precision settings from §4.

## 7. Shared tests and documentation

**Duplication measured:** line diffs after normalising `nerva_<backend>` → `nerva_X`.
- **Identical across all four backends:** `test_datasets_basics.py`, `test_learning_rate_schedulers.py`, `run_all_tests.py`, `tools/mlp.py`, `inspect_npz.py`, `train_mnist.py`, `run_all_examples.py`, `custom_components_validation.py`.
- **Small diffs (≤ 50 lines):** `test_loss_functions`, `test_matrix_operations_basics`, `test_multilayer_perceptrons`, `test_one_layer`, `test_optimizers_basics`, `test_softmax_functions_basics`.
- **Real divergence:**
  - `tests/utilities.py` (≈100 lines)
  - `matrix_operations.py`
  - `weight_initializers.py`: JAX uses a functional `weights_*` API returning new arrays, while the others use an in-place `set_weights_*` API
  - `layers.py` and `optimizers.py` (JAX/TF immutability)

**Proposal:**
- **Shared conformance test suite.** Add `nerva_sympy.testing` to nerva-sympy; it needs only numpy and sympy. The tests are parametrised over a small **backend adapter**:
  - `from_numpy` and `to_numpy`
  - the `Matrix` type
  - weight-setting helpers that hide the in-place vs functional difference
  - module handles
  Each backend repo keeps a ~20-line adapter plus a `conftest.py` that runs the shared suite against its own framework. Backend-specific tests stay local.
- **Shared documentation.** The four READMEs are already near-identical templates. Generate them, plus shared AsciiDoc chapters such as the matrix-operations table and the install matrix, from one template with per-backend variables. Turn the docs deploy jobs into one reusable GitHub workflow.

## 8. nerva-sympy without all four frameworks (special focus)

**Key finding: the coupling is accidental.**
- `src/nerva_sympy/*` (1,493 lines) imports **only `sympy` and `numpy`**. The four backends are needed only by some tests and scripts.
- `pyproject.toml` nevertheless lists all four backends as hard runtime dependencies.
- **`tests/utilities.py` imports `jax`, `tensorflow` and `torch` at module level.** As a result, every test that does `from utilities import …` needs all three frameworks, even the purely symbolic ones. That covers about 13 of the ~25 test files: `test_derivatives`, `test_layer_*`, `test_*_derivations`, `test_matrix_properties`, `test_batch_normalization_definition` and others.
- **The equation-consistency checks need no frameworks at all.** `scripts/check-equations-<backend>.py` parse the backend's `layers.py` source as text/AST and match it against the `# feedforward` / `# backpropagation` blocks in the SymPy tests. Today they find that source through a hard-coded sibling checkout (`../nerva-<backend>/src/...`).
- **The pytest wrappers for those checks are never collected.** They are named `tests/test-equations-*.py` (hyphens), while pytest's default pattern and the `unittest` fallback in `run_all_tests.py` both match `test_*.py` only.
- The backend packages' `__init__.py` files import nothing. So `importlib.util.find_spec("nerva_jax")` can locate the installed `layers.py` without importing JAX, and even when the backend was installed with `--no-deps`.

**Proposed tiered design:**

| Tier | What it validates | Needs | How |
|---|---|---|---|
| 0: symbolic core | Matrix-form equations against SymPy differentiation | `sympy`, `numpy` | Make the framework imports in `tests/utilities.py` lazy (move `to_torch`/`to_jax`/`to_tensorflow` behind on-demand imports). The ~13 symbolic tests then run with only `pip install nerva-sympy` |
| 1: equation text | The backend's `layers.py` equations are textually identical to the validated SymPy blocks | backend *source* only (checkout or `--no-deps` install) | Rename to `test_equations_*.py`. Resolve the source via an env var or `find_spec`, falling back to the sibling path. Runs for all four backends in one lightweight job |
| 2: numerical, per backend | The backend's numbers match SymPy and the golden fixtures | one framework | `pytest.importorskip("torch")` etc. Extras `nerva-sympy[numpy]`, `[jax]`, `[tensorflow]`, `[torch]` |
| 3: all frameworks together | Full cross-framework agreement (`test_frameworks.py`, `test_one_hot.py`) | `nerva-sympy[all]` | Optional nightly job with a fixed seed |

Resulting `pyproject.toml` (proposal):
- `dependencies = ["numpy", "sympy"]`
- extras: one per backend, plus `all` and `test`

Tier 1 also removes the "monorepo sibling layout" assumption, which today silently skips the checks when the siblings are absent.

---

## 9. Other defects found (Python repos; for the backlog)

**CLI and example scripts:**
- **`tools/mlp.py` (all four backends):**
  - `--layer-sizes` help text and default say comma-separated (`'3072,128,64,10'`), but the code splits on `;`, so the default crashes.
  - `--layers` and `--layer-weights` have no default, so omitting them raises `AttributeError` on `None.split`.
  - There is no `--seed` option.
- **`examples/train_mnist_cli.sh`:**
  - `PYTHONPATH=../src` is assigned but never exported, so the script only works if the package is already installed.
  - It refers to `prepare_datasets.py`, but the script is called `prepare_data.py`.
- **`datasets.create_npz_dataloaders(filename, batch_size: int=True)`:** the default value is a bool.

**Docstring copy-paste errors:**
- `nerva_numpy/__init__.py` says "built on top of PyTorch".
- `nerva_tensorflow/utilities.set_tensorflow_options` says "Configure PyTorch…".

**Test configuration:**
- The `slow` pytest marker is used but never registered, and there is no `[tool.pytest.ini_options]`.
- Running the backends' validation examples needs nerva-sympy, which installs all the other backends. This circularity is resolved by §8.

---

## 10. Proposed execution plan (nothing starts without approval)

**Phase A: confirm the baseline (read-only, scratch environments outside the repos).** In throwaway venvs, using `uv` for speed, for Python 3.11, 3.12, 3.13 and 3.14:
- Install each backend from PyPI 1.0.0 and from source, then run its tests. Record the results, the resolved versions, the install size and the time.
- Install nerva-sympy 1.0.0 to measure the cost and conflicts of the current all-framework install.
- Try torch CPU-index vs default installs, `jax[cpu]` vs `jax[cuda13]` (resolution only), and `tensorflow-cpu`.
- Run the 4 equation-check scripts with sibling checkouts, then run `test_frameworks.py` with a fixed seed and the data generated by `prepare_data.py`.

Output: a compatibility matrix (backend × Python × framework version, pass/fail) and a list of measured conflicts. This answers questions 1–4 empirically.

**Phase B: proposed PR backlog.** Small and independently reviewable, roughly in this order:
1. **nerva-sympy decoupling:** lazy framework imports in test utilities; rename the equation tests and make their source path configurable; core dependencies cut to sympy+numpy with per-backend extras; commit the golden one-layer cases; give `test_frameworks` a fixed default seed.
2. **Test CI for each backend:** Python 3.11–3.14 matrix, CPU wheels (torch via the CPU index), and the Tier 0/1 nerva-sympy jobs.
3. **Packaging cleanup** (§5) with tested lower bounds, plus a minimum-versions CI job.
4. **CLI and script bug fixes** (§9).
5. **Golden fixtures and the shared conformance suite** (§6–7), rolled out one backend at a time.
6. **Install documentation:** a per-platform CPU/GPU table per backend, and a correction to the README's GPU limitation statement for the Python backends.
7. **Templated READMEs and docs,** a reusable docs workflow, and removing the torch install from the docs build.

**Explicitly out of scope:** alternative C++ backends, and making MKL optional. The C++ repos keep MKL as the performance backend; they get only incidental documentation fixes if requested.

## Verification
- **Phase A:** the compatibility matrix is produced by scripted runs. Each cell links to its log.
- **Each Phase B PR:** the repo's own tests must pass in CI on the agreed Python matrix.
- **nerva-sympy decoupling:** in a venv with **only** `pip install -e nerva-sympy` (numpy and sympy), the Tier 0 and Tier 1 tests must pass. In a venv with one backend, that backend's Tier 2 tests must pass. In the `[all]` environment, the full suite must give the same results as today.
- **Consistency:** each backend must match the committed golden fixtures within the documented float32 tolerances. The nightly all-framework job must stay green with a fixed seed.
