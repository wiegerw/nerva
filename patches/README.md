# Patches for main.tex

## batch-normalization.patch

Removes the transposed diagonal `diag(.)^T` from the batch normalization equations. In the Python
backends `diag` returns a 1-D array, so the transpose in `diag(R.T @ R).T` has no effect, and PyTorch
has deprecated `.T` on 1-D tensors. The new equations use column sums instead:

| Old | New | Code |
|-----|-----|------|
| `diag(R^T R)^T` | `1_N^T (R ⊙ R)` | `columns_sum(hadamard(R, R))` |
| `diag(Z^T ΔZ)^T` | `1_N^T (Z ⊙ ΔZ)` | `columns_sum(hadamard(Z, DZ))` |

The new form is also cheaper: it avoids computing the D×D matrix `R^T R`. It matches the equation
for `Δγ = 1_N^T (Z ⊙ ΔY)` that is already used.

The patch changes:

1. the preamble: a macro `\bnchange{...}` that highlights the changes in red;
2. the Batch normalization section: the text that defines Σ, and the feedforward equation for Σ;
3. Appendix, batch normalization layer: the feedforward equation for Σ, the backpropagation equation
   for ΔX, and both code listings;
4. matrix properties 1 and 2, which are used to derive the matrix form of ΔX, plus one new sentence
   after the properties that relates `1_m^T (X ⊙ Y)` to `diag(X^T Y)^T`;
5. Appendix, derivation of the batch normalization layer: the final matrix-form equation for ΔX.

The softmax equations, matrix properties 5 and 6, and the `diag` row of the implementation table are
unchanged: they use `diag` as a column vector without a transpose.

The new equations are the same as the ones in nerva-sympy and in the four Python backends.

### How to apply

Run these commands from the root of this branch:

```bash
git apply --check patches/batch-normalization.patch   # dry run
git apply patches/batch-normalization.patch
```

Alternatively, `patch -p1 < patches/batch-normalization.patch` (use `--dry-run` to test first).
To revert the patch: `git apply -R patches/batch-normalization.patch`.

### How to inspect the changes

- In the PDF, every changed formula and sentence is shown in red. Add `\showbnpatchfalse` after the
  macro definition in the preamble to switch the colors off.
- In the source, every change is preceded by a comment line that starts with `%% BN-PATCH` and
  contains the old text (12 in total):

  ```bash
  grep -n "BN-PATCH" main.tex
  ```

  The code listings (`lstlisting`) cannot contain macros, so these changes only have the comment.

### How to remove the markers afterwards

1. Delete all lines that start with `%% BN-PATCH`.
2. Replace every `\bnchange{X}` by `X`.
3. Delete the `\newif\ifshowbnpatch` and `\newcommand{\bnchange}` lines from the preamble.
