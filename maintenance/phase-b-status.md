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
**Partly.** Items 1–4, 6 and 7 are done; item 5 is partly done.

| # | Phase B backlog item | Status |
|---|---|---|
| 1 | nerva-sympy decoupling: lazy imports, `test_equations.py`, optional backends, golden one-layer cases, fixed seed and smooth loss in `test_frameworks.py` | ✅ Done (nerva-sympy PR). The 5 slow tests are also marked |
| 2 | Test CI for each backend, plus nerva-sympy CI | ✅ Done (all 5 PRs; all green) |
| 3 | Packaging cleanup with tested minimum versions, plus a minimum-versions CI job | ✅ Done (backend PRs) |
| 4 | CLI and script bug fixes (`mlp.py`, `train_*_cli.sh`, docstrings), plus `--seed` | ✅ Done (backend PRs) |
| 5 | Golden fixtures and a **shared conformance test suite** across the backends | ◐ Partly done. The golden one-layer cases are shared and pass in all backends. The shared suite (one set of tests, a small adapter per backend) is not done |
| 6 | Install documentation; correct the umbrella README's "no GPU support" statement | ✅ Done. Backend READMEs have CPU/GPU notes; the umbrella README states the GPU support precisely and has a "Which repository should I use?" section |
| 7 | Templated READMEs/docs, a reusable docs workflow, no torch install in the docs build | ◐ The nerva-torch docs workflow no longer installs torch (docs build verified without it). Templated READMEs and a reusable docs workflow are not done; nerva-jax/-tensorflow build docs in custom container images |

## Other open items, from earlier findings
- **nerva-torch `layers.py:261`:** the `diag(R.T @ R).T` deprecation warning. It needs your decision, because the fix changes the equation text that `test_equations.py` checks.
- **nerva-sympy cross-framework tests:** they still need all four frameworks. The rewrite to compare one backend at a time is not done.
- **Umbrella repo `wiegerw/nerva`:** done on branch `claude/dazzling-fermat-qi376i` (LICENSE added, README updated, `data/prepare_data.py` made robust). The fixed `prepare_data.py` is synced to the five Python repos' branches. nerva-rowwise still has the old copy.
- **Correction to the Stage 2 report:** the old `prepare_data.py` did *not* produce float64 CIFAR-10 data; it converts to float32 before dividing. The new script produces identical output.
- **nerva-colwise and nerva-rowwise (C++):** only documentation fixes were proposed. They are out of scope for now.

## Next step
Item 5: a shared conformance test suite for the four backends. This needs a design decision (see the chat).
