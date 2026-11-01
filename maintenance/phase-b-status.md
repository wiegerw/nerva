# Nerva maintenance: where things are, and Phase B status

## Where to find things
- **Phase B backlog, the original plan:** §10 of `stage2-investigation.md` in this folder.
- **Phase A results:** `phase-a-report.md` in this folder.
- **The code changes** are in the five pull requests on GitHub:
  - https://github.com/wiegerw/nerva-sympy/pull/1
  - https://github.com/wiegerw/nerva-numpy/pull/1
  - https://github.com/wiegerw/nerva-jax/pull/1
  - https://github.com/wiegerw/nerva-tensorflow/pull/1
  - https://github.com/wiegerw/nerva-torch/pull/1

  Each PR page has three tabs:
  - **Conversation:** the description.
  - **Commits:** the individual commits.
  - **Files changed:** the diff.

  The green check marks show the CI results. Merging is up to you, with the "Merge pull request" button at the bottom of the Conversation tab.

## Is Phase B finished?
**Partly.** The first four backlog items are done and in the PRs. Items 5–7 are not done yet.

| # | Phase B backlog item | Status |
|---|---|---|
| 1 | nerva-sympy decoupling: lazy imports, `test_equations.py`, optional backends, golden one-layer cases, fixed seed and smooth loss in `test_frameworks.py` | ✅ Done (nerva-sympy PR). The 5 slow tests are also marked |
| 2 | Test CI for each backend, plus nerva-sympy CI | ✅ Done (all 5 PRs; all green) |
| 3 | Packaging cleanup with tested minimum versions, plus a minimum-versions CI job | ✅ Done (backend PRs) |
| 4 | CLI and script bug fixes (`mlp.py`, `train_*_cli.sh`, docstrings), plus `--seed` | ✅ Done (backend PRs) |
| 5 | Golden fixtures and a **shared conformance test suite** across the backends | ◐ Partly done. The golden one-layer cases are shared and pass in all backends. The shared suite (one set of tests, a small adapter per backend) is not done |
| 6 | Install documentation; correct the umbrella README's "no GPU support" statement | ◐ Partly done. The backend READMEs have CPU/GPU notes. The umbrella README (`wiegerw/nerva`) is not updated |
| 7 | Templated READMEs/docs, a reusable docs workflow, no torch install in the docs build | ❌ Not started |

## Other open items, from earlier findings
- **nerva-torch `layers.py:261`:** the `diag(R.T @ R).T` deprecation warning. It needs your decision, because the fix changes the equation text that `test_equations.py` checks.
- **nerva-sympy cross-framework tests:** they still need all four frameworks. The rewrite to compare one backend at a time is not done.
- **Umbrella repo `wiegerw/nerva` (from the Stage 1 report):**
  - LICENSE file missing; the badge link is broken.
  - `data/prepare_data.py` defects (float64 CIFAR, `tarfile` without a filter, no timeouts/checksums, exit code 0 on errors).
  - README entry-point improvements.
- **nerva-colwise and nerva-rowwise (C++):** only documentation fixes were proposed. They are out of scope for now.

## Proposed next step, after your approval
Continue Phase B with the remaining items, in this order:
1. **Umbrella repo `wiegerw/nerva`:**
   - Add `LICENSE`.
   - Fix `prepare_data.py`.
   - Correct the GPU statement and add a "which repo to use" section to the README.

   This finishes item 6.
2. **Item 7:** remove the unneeded torch install from the nerva-torch docs workflow.
3. **Item 5 (larger):** a shared conformance suite in nerva-sympy, with an adapter for each backend.

Each step is its own branch and PR. Nothing starts until you approve.
