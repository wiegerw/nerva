# Nerva CNN primitives — specification (revision 3)

*Status: proposal for review. Supersedes §2.5, §3 and the primitive table of §4 in `design-report.md` (revision 2). No repository has been changed.*

## 0. Reshape is no longer used

You are right about revision 2's `reshape(X, m, n)`. It keeps the sequence of elements in row-major order and only changes the number of rows and columns. That is a statement about how the numbers are stored in memory, not an operation of matrix algebra.

In matrix calculus the same thing appears as `vec`/`unvec`, so it can be made rigorous. But it does not belong next to `hadamard` and `columns_sum`, and it makes the equations harder to read.

**Revision 3 removes `reshape` from the mathematics completely.** Two changes make this possible:

1. **Feature maps are stored as a pixel matrix.** There is one row per pixel and one column per channel (§1). The output of a convolution then naturally has this form, so nothing has to be reinterpreted.
2. **Every new operation is defined by multiplication with explicit 0/1 selection matrices** `S_pq` (§3), and by splitting a matrix into column blocks (§5).

The only part that cannot be expressed this way is a pointwise maximum, which pooling needs. That is the same situation as the existing `rows_max`.

All definitions below were checked numerically with a throwaway prototype in the scratchpad. In particular, it builds `S_pq` explicitly and confirms that the efficient implementation equals the matrix definition exactly.

---

## 1. Representation of a batch of images

A batch of `N` images of height `H`, width `W` and `C` channels is the **pixel matrix**

```
X ∈ ℝ^{(N·H·W) × C}
```

- Each row is one pixel of one image. Each column is one channel.
- The pixels are numbered image by image, row by row:
  `row(n, h, w) = (n·H + h)·W + w`, for `0 ≤ n < N`, `0 ≤ h < H`, `0 ≤ w < W`.
- Element `X[row(n,h,w), c]` is the value of channel `c` at pixel `(h, w)` of image `n`.

**Example.** A batch of `N = 1` RGB image of size 2×2 (`C = 3`) is a 4×3 matrix:

```
            R    G    B
(h,w)=(0,0) x00  x01  x02
(h,w)=(0,1) x10  x11  x12
(h,w)=(1,0) x20  x21  x22
(h,w)=(1,1) x30  x31  x32
```

This plays the same role for images that "one row per sample" plays for the existing MLP layers. Between convolution and pooling layers, a batch is a pixel matrix. Between dense layers it stays an `N × D` matrix, as today.

**Facts about the existing code that this affects** [F]:
- `prepare_data.py:170-172` stores CIFAR-10 as one row per image of 3072 values in (h, w, c) order.
- The `DataLoader` already converts data per batch: it one-hot encodes the labels (`datasets.py`).
- The training loop divides the loss gradient by `X.shape[0]` (`training.py:157`).

**Consequences** [R]:
- The loader (or a one-off conversion when the data is loaded) delivers image batches as pixel matrices. This is a change of data layout, done once outside the layers.
- For a batch of images, `X.shape[0]` is `N·H·W`, so the normalisation in `training.py:157` must use the number of samples `N`. For example it can use `Y.shape[0]`, because the network output is still `N × K`.

---

## 2. Geometry

A convolution or pooling window is described by a geometry record:

```
g = (N, H, W, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr)
```

| Symbol | Meaning |
|---|---|
| `N, H, W` | batch size, input height and width |
| `kh, kw` | window (kernel) height and width |
| `sh, sw` | stride: how far the window moves per output pixel (default 1) |
| `dh, dw` | dilation: distance between the window's sample points (default 1) |
| `pt, pb, pl, pr` | number of zero rows/columns added at the top, bottom, left and right (default 0) |

The output height and width are

```
H′ = ⌊(H + pt + pb − dh·(kh−1) − 1) / sh⌋ + 1
W′ = ⌊(W + pl + pr − dw·(kw−1) − 1) / sw⌋ + 1
```

and `M = N·H′·W′` is the number of output pixels. The geometry contains no channel count: the operations below work for any `C`.

---

## 3. Selection matrices `S_pq`

For every position `(p, q)` inside the window (`0 ≤ p < kh`, `0 ≤ q < kw`) there is a 0/1 matrix

```
S_pq ∈ {0,1}^{M × (N·H·W)}
```

defined as follows. Row `(n,i,j)` of `S_pq` corresponds to output pixel `(i,j)` of image `n`. Let

```
h = sh·i + dh·p − pt,    w = sw·j + dw·q − pl.
```

- If `0 ≤ h < H` and `0 ≤ w < W`, the row contains a single 1, in column `row(n,h,w)`.
- Otherwise the window position falls in the padding, and **the row is entirely zero**.

So `S_pq · X` is an `M × C` matrix. Its row `(n,i,j)` is the pixel that window position `(p,q)` reads when the window is placed for output pixel `(i,j)`, or zero in the padding. Multiplying by `S_pq` only selects and copies rows of `X`.

**Example.** `N = 1`, 3×3 image, 2×2 window, stride 1, no padding, so `H′ = W′ = 2` and `M = 4`. The input pixels are numbered 1…9 row by row:

```
image      S_01  (window position p=0, q=1: "top right")
1 2 3      [0 1 0 0 0 0 0 0 0]   output (0,0) reads pixel 2
4 5 6      [0 0 1 0 0 0 0 0 0]   output (0,1) reads pixel 3
7 8 9      [0 0 0 0 1 0 0 0 0]   output (1,0) reads pixel 5
           [0 0 0 0 0 1 0 0 0]   output (1,1) reads pixel 6
```

Stride, dilation and padding change *which* column holds the 1, or make a row zero. The structure does not change.

The matrices `S_pq` are a **mathematical device**: no implementation ever builds them, because they are extremely sparse.

---

## 4. `im2col` and `col2im`

### 4.1 `im2col(X, g)`: collect the window contents

For `X ∈ ℝ^{(N·H·W)×C}`:

```
im2col(X, g) = [ S_00·X   S_01·X   …   S_(kh−1)(kw−1)·X ]   ∈ ℝ^{M × (kh·kw·C)}
```

This is a horizontal concatenation of `kh·kw` blocks, ordered `p` first and then `q`. Each block is `M × C`. Row `(n,i,j)` of the result contains everything the window sees when placed at output pixel `(i,j)`: first the C channels at position (0,0), then the C channels at position (0,1), and so on.

**Example** (the 3×3 image above, `C = 1`):

```
              (0,0) (0,1) (1,0) (1,1)   ← window position
output (0,0) [  1     2     4     5  ]
output (0,1) [  2     3     5     6  ]
output (1,0) [  4     5     7     8  ]
output (1,1) [  5     6     8     9  ]
```

`im2col` only copies values; it does no arithmetic. It is linear in `X`.

### 4.2 `col2im(P, g)`: return contributions to the pixels, with summation

For `P = [P_00  P_01  …] ∈ ℝ^{M × (kh·kw·C)}`, split into the same `kh·kw` blocks of width `C`:

```
col2im(P, g) = Σ_{p=0}^{kh−1} Σ_{q=0}^{kw−1}  S_pqᵀ · P_pq    ∈ ℝ^{(N·H·W) × C}
```

`S_pqᵀ` sends row `(n,i,j)` of `P_pq` back to the pixel it was read from. A pixel that is seen by several windows receives the **sum** of all the corresponding entries. Entries that came from padding are dropped, because the zero rows of `S_pq` become zero columns of `S_pqᵀ`. This is the summation that convolution needs.

**Example.** For the 3×3 image, `col2im` of the 4×4 all-ones matrix is

```
1 2 1
2 4 2
1 2 1
```

shown as an image. Each pixel gets the number of window positions that cover it. The centre pixel 5 is covered four times, and its value is `P[1,4] + P[2,3] + P[3,2] + P[4,1]` (1-based indices).

**Key property.** `col2im` is the adjoint (transpose) of `im2col`:

```
elements_sum(hadamard(im2col(X), P)) = elements_sum(hadamard(X, col2im(P)))
```

This follows directly from the definitions, and it is exactly why `col2im` appears in backpropagation.

---

## 5. Block operations

Let `P = [P_1 P_2 … P_B]` be a matrix of `B` blocks of equal width `C` (so `P` has `B·C` columns). For pooling, `B = kh·kw`.

| Code | Definition | Size |
|---|---|---|
| `blocks_sum(P, B)` | `P_1 + P_2 + … + P_B` | `M × C` |
| `blocks_repeat(Y, B)` | `[Y Y … Y]` (`B` copies side by side) | `M × B·C` |
| `blocks_max(P, B)` | elementwise maximum `max(P_1, …, P_B)` | `M × C` |
| `blocks_max_mask(P, B)` | `[E_1 … E_B]`, where `E_b[r,c] = 1` if block `b` is the **first** block in which `P_b[r,c]` equals the maximum, and 0 otherwise | `M × B·C` |

**How they relate to the existing table:**
- For `C = 1` they reduce exactly to existing operations: `blocks_sum = rows_sum`, `blocks_repeat = column_repeat`, `blocks_max = rows_max`. They are the natural generalisation to blocks.
- `blocks_sum` and `blocks_repeat` are ordinary matrix products: `blocks_sum(P,B) = P·(1_B ⊗ I_C)` and `blocks_repeat(Y,B) = Y·(1_Bᵀ ⊗ I_C)`, where `1_B` is the `B × 1` vector of ones. In the same way, `row_repeat(x, m) = 1_m·x`.

**Example** (`B = 2`, `C = 2`):

```
P = [ 1 5 | 3 5 ]      blocks_max(P,2) = [ 3 5 ]      blocks_max_mask(P,2) = [ 0 1 | 1 0 ]
    [ 4 0 | 2 0 ]                        [ 4 0 ]                             [ 1 1 | 0 0 ]
```

The `5` in the first row and the `0` in the second row are ties; the first block receives the 1.

---

## 6. Layers expressed in the primitives

Below, `@` is the matrix product. All layer code is the same in every backend.

### 6.1 Convolution

The weights are `W = [W_00 W_01 … W_(kh−1)(kw−1)] ∈ ℝ^{K × (kh·kw·C)}`, with `W_pq ∈ ℝ^{K×C}` the weights of window position `(p,q)`. The bias is `b ∈ ℝ^K`. The mathematical definition is

```
Z = Σ_{p,q} S_pq·X·W_pqᵀ + 1_M·bᵀ = im2col(X, g)·Wᵀ + row_repeat(b, M)
```

In code:

```
# feedforward
Xc = im2col(X, g)
Z  = Xc @ W.T + row_repeat(b, M)
Y  = act(Z)

# backpropagation
DZ = hadamard(DY, act.gradient(Z))
DW = DZ.T @ Xc
Db = columns_sum(DZ)
DX = col2im(DZ @ W, g)
```

These are exactly the `LinearLayer`/`ActivationLayer` equations with `X` replaced by `Xc`. The only new step is the last one.

**Derivation of the last line.**
- `Z` depends on `X` through `Σ_{p,q} S_pq·X·W_pqᵀ`.
- So `DX = Σ_{p,q} S_pqᵀ·DZ·W_pq`.
- `DZ @ W = [DZ·W_00  DZ·W_01  …]`, so this is `col2im(DZ @ W, g)` by definition.

The input `X` is `(N·H·W) × C` and the output `Z` is `M × K = (N·H′·W′) × K`. Both are pixel matrices, so the next layer can be a convolution or pooling layer directly.

### 6.2 Max pooling

Here `B = kh·kw`. There are no parameters and no padding in version 1, because zero padding would be wrong for a maximum.

```
# feedforward
P = im2col(X, g)
Y = blocks_max(P, B)

# backpropagation
DX = col2im(hadamard(blocks_max_mask(P, B), blocks_repeat(DY, B)), g)
```

The gradient of each output pixel goes to the first input pixel that attained the maximum in its window. Pixels covered by several windows (stride smaller than the window) receive the sum.

### 6.3 Average pooling

```
# feedforward
Y = blocks_sum(im2col(X, g), B) / B

# backpropagation
DX = col2im(blocks_repeat(DY, B), g) / B
```

### 6.4 From feature maps to dense layers ("flatten")

Take the geometry `g_full` with a window as large as the image: `kh = H`, `kw = W`, stride 1, no padding. Then `H′ = W′ = 1`, `M = N`, and

```
im2col(X, g_full) ∈ ℝ^{N × (H·W·C)}
```

has one row per image, containing all its pixels in (h, w, c) order. This is precisely the existing representation of the dense layers, and the same order as the CIFAR-10 file. Hence:

```
# feedforward                     # backpropagation
Y = im2col(X, g_full)             DX = col2im(DY, g_full)
```

No new operation is needed. Equivalently, "flatten followed by a linear layer" is the same as a convolution with a full-image window.

### 6.5 Bonus: batch normalisation of feature maps

The existing `BatchNormalizationLayer` normalises each column over all rows. Applied to a pixel matrix, that means each channel is normalised over all pixels of all images. That is standard spatial batch normalisation, with no new equations.

---

## 7. CIFAR-10 example in this representation (batch size N = 100)

| Stage | Matrix | Size |
|---|---|---|
| input | pixel matrix | 102,400 × 3 |
| Conv 3×3, K=32, pad 1 | `Xc`, `Z` | 102,400 × 27, 102,400 × 32 |
| MaxPool 2×2, stride 2 | `P`, `Y` | 25,600 × 128, 25,600 × 32 |
| Conv 3×3, K=64, pad 1 | `Xc`, `Z` | 25,600 × 288, 25,600 × 64 |
| MaxPool 2×2, stride 2 | `P`, `Y` | 6,400 × 256, 6,400 × 64 |
| Flatten (`g_full`, 8×8) | `Y` | 100 × 4,096 |
| Linear | `Y` | 100 × 10 |

The parameters are the same as in the report (60,362).

---

## 8. Implementation notes (not part of the mathematics)

- `S_pq` is never built.
  - `im2col` is implemented by `kh·kw` strided slice copies.
  - `col2im` by `kh·kw` strided slice additions.
  - The block operations by column slicing.
- Inside these implementations, viewing the pixel matrix's memory as an `N×H×W×C` array is a free and convenient *implementation technique*. It never appears in a layer equation.
- Each framework implements the primitives with its own calls (NumPy and `jnp` slices or `.at[].add`; `tf.image.extract_patches` / `tf.math.unsorted_segment_sum`; `F.unfold` / `F.fold`). Only the column order may need adapting.
- nerva-sympy can implement the primitives *literally* from §3–§5, building `S_pq` as an `sp.Matrix`. This is feasible for the tiny sizes used in symbolic tests, and it makes the reference implementation a direct transcription of this specification.

**Prototype results** [P]:
- the efficient `im2col`/`col2im` equal the `S_pq` definitions exactly (difference 0);
- convolution equals a direct seven-loop implementation (≤ 4e-15);
- `DX`, `DW` and `Db` of convolution and `DX` of both pooling layers agree with finite differences (≤ 1e-8), for strides 1–3, dilation 2, asymmetric padding, non-square windows and overlapping pooling windows;
- flatten via `g_full` equals the (h, w, c) row-per-image layout, and `col2im` inverts it exactly.

---

## 9. Can the S_pq model be implemented efficiently?

Yes, **as long as `S_pq` is never stored as a matrix.** The `S_pq` are a way to *specify* `im2col` and `col2im`. Their structure is so regular that computing a product with them is just a strided copy. Three ways to compute `S_pq·X`, from worst to best:

| Approach | What it does | Cost (CIFAR conv1, N = 100) |
|---|---|---|
| Dense `S_pq`, ordinary matrix product | literally multiply by a 102,400 × 102,400 matrix | about **352 GiB** of memory for the nine conv1 matrices (22 GiB for conv2): impossible |
| Sparse `S_pq` (index tables) | each row of `S_pq` has at most one 1, so `S_pq·X` is a *gather* of rows of `X` and `S_pqᵀ·P` is a *scatter-add* | works; in NumPy gather 10–18 ms, scatter-add (`np.add.at`) 39–74 ms per layer [P] |
| **Strided slices (recommended)** | the 1s of `S_pq` lie on an affine pattern (`h = sh·i + dh·p − pt`, `w = …`). So `S_pq·X` is a strided slice of the zero-padded pixel matrix (viewed as an N×H×W×C array), and `S_pqᵀ·P` is a strided slice *addition* | `im2col` ≈ 6 ms, `col2im` ≈ 10–16 ms per layer [P] |

### 9.1 Cost model
- For a convolution with `K` output channels, `im2col` and `col2im` each move `M·kh·kw·C` numbers. The matrix products perform `2·M·kh·kw·C·K` floating-point operations, three times per training step (`Z`, `DW`, `DX`).
- The data movement is therefore about `1/K` of the arithmetic work, but it is memory-bound, while the matrix products run at BLAS speed.
- [P] NumPy measurements, N = 100, float32, 4 cores, ms per layer:

  | Layer | `im2col` | Product for `Z` | Products for `DW`, `DX` | `col2im` |
  |---|---|---|---|---|
  | conv1 (32×32×3 → 32) | 6 | 1.1 | 5.3 | 16 |
  | conv2 (16×16×32 → 64) | 6 | 2.9 | 6.3 | 10 |

  Data movement and arithmetic are of the same order. The `col2im` of the first layer can be skipped, because no `DX` is needed for the input.
- For the whole CIFAR network, a NumPy training step is about 100–130 ms per batch of 100. That is roughly **1 minute per epoch**, which is acceptable for the NumPy backend.

### 9.2 Per framework
The same strided-slice algorithm works everywhere. Some frameworks also have a ready-made kernel [E]:

| | `im2col` (`S_pq·X`) | `col2im` (`Σ S_pqᵀ·P_pq`) |
|---|---|---|
| NumPy | `kh·kw` strided slice copies | `kh·kw` strided slice `+=` |
| PyTorch | `F.unfold` (native C++/CUDA kernel) | `F.fold` (native), which is exactly the adjoint of `unfold` |
| JAX | `lax.conv_general_dilated_patches`, or slices | `x.at[slices].add(…)` per `(p,q)`; XLA fuses these |
| TensorFlow | `tf.image.extract_patches`, or slices + `tf.concat` | `tf.math.unsorted_segment_sum` with an index table (the sparse form above). TF has no strided slice-add |

The block operations are column slices plus `maximum`/`argmax`, which every framework has.

### 9.3 What the modelling costs, compared with native convolution
- **Memory:** `Xc = im2col(X)` has `kh·kw` times as many entries as `X` (stride 1): 10.5 MiB for conv1 and 28 MiB for conv2 at N = 100. This is fine for CIFAR-sized networks. For very large feature maps it is the main limitation, and the remedy (processing the batch in chunks) stays inside the implementation.
- **GPU speed:** native cuDNN kernels avoid building `Xc` and can use faster algorithms. JAX, TensorFlow and PyTorch on GPU will therefore be slower than with their built-in convolution, probably a few times [A]. On CPU the difference is smaller.
- **The representation is free:** the pixel matrix is the same memory buffer as an NHWC array, so no conversions are needed between layers.

## 10. Combining operations into larger expressions

### 10.1 Where the elementary formulation costs performance

| Layer | Elementary formulation | Penalty compared with a framework's native kernel |
|---|---|---|
| Convolution | `Xc = im2col(X)`, then matrix products, then `col2im` | **Memory:** `Xc` has `kh·kw` times as many numbers as `X` (×9 for 3×3), and it is kept until the backward pass. **GPU speed:** native kernels (cuDNN, oneDNN) never build `Xc`, and use better algorithms. Expected to be a few times faster on GPU [A, not measured]. |
| Max / average pooling | `im2col`, then block operations | Small. For the usual 2×2 windows with stride 2, `im2col` copies every number exactly once (no duplication). |
| Flatten | `im2col` with a full-image window | None: it is a single copy. |

**Conclusion:** only the convolution layer is worth combining. Its three matrix expressions each correspond to one well-known native kernel.

### 10.2 Three combined operations, each defined by a primitive expression
Each combined operation is defined by one equation in the primitives of §3–§5, and additionally by an index formula. The index formulas make each operation readable without the selection matrices.

Notation: `X[(n,h,w), c]` is pixel `(h,w)` of image `n`, channel `c`. A term whose pixel index falls outside the image counts as 0.

**(F1) Correlation:** the forward pass, `ℝ^{(N·H·W)×C} × ℝ^{K×(kh·kw·C)} → ℝ^{M×K}`.
```
correlate(X, W, g) := im2col(X, g) · Wᵀ

correlate(X, W, g)[(n,i,j), k] = Σ_{p,q,c}  W[k,(p,q,c)] · X[(n, sh·i+dh·p−pt, sw·j+dw·q−pl), c]
```

**(F2) Correlation, weight gradient:** `ℝ^{M×K} × ℝ^{(N·H·W)×C} → ℝ^{K×(kh·kw·C)}`.
```
correlate_weight_gradient(DZ, X, g) := DZᵀ · im2col(X, g)

correlate_weight_gradient(DZ, X, g)[k,(p,q,c)] = Σ_{n,i,j}  DZ[(n,i,j),k] · X[(n, sh·i+dh·p−pt, sw·j+dw·q−pl), c]
```

**(F3) Correlation, input gradient:** `ℝ^{M×K} × ℝ^{K×(kh·kw·C)} → ℝ^{(N·H·W)×C}`.
```
correlate_input_gradient(DZ, W, g) := col2im(DZ · W, g)

correlate_input_gradient(DZ, W, g)[(n,h,w), c] = Σ_{(i,p): sh·i+dh·p−pt = h}  Σ_{(j,q): sw·j+dw·q−pl = w}  Σ_k  DZ[(n,i,j),k] · W[k,(p,q,c)]
```

The names are placeholders (§11). With these operations the convolution layer reads:
```
# feedforward
Z  = correlate(X, W, g) + row_repeat(b, M)
Y  = act(Z)

# backpropagation
DZ = hadamard(DY, act.gradient(Z))
DW = correlate_weight_gradient(DZ, X, g)
Db = columns_sum(DZ)
DX = correlate_input_gradient(DZ, W, g)
```

### 10.3 Why this keeps the mathematics clean
- **Mathematically, the two versions of the layer are identical.** The combined version only replaces each expression by its defining name.
  - nerva-sympy proves F1–F3 equal their definitions.
  - The tests check that every backend's implementation of F1–F3 agrees with the elementary version.
- **The implementation is free per framework:**
  - NumPy implements F1–F3 literally by their definitions: `im2col` + BLAS, so it loses nothing.
  - PyTorch can use `F.conv2d`, `torch.nn.grad.conv2d_weight` and `torch.nn.grad.conv2d_input` [E].
  - TensorFlow can use `tf.nn.conv2d`, `Conv2DBackpropFilter` and `Conv2DBackpropInput` [E].
  - JAX can use `lax.conv_general_dilated`, and the same operation with the roles of the arguments exchanged for F2 and F3 [E].

  None of these build `Xc`, and F2 can work from `X` directly, so `Xc` no longer has to be stored between the forward and the backward pass.
- **The documentation shows both levels:** the combined layer equations, and below them the one-line definitions of F1–F3 in terms of `im2col`, `col2im` and the matrix product.

### 10.4 Recommendation
1. Specify both levels now: the primitives of §3–§5 and the combined operations F1–F3.
2. Implement the elementary version first (milestones M4–M6), because it is the easiest to verify.
3. Then measure PyTorch and JAX on CPU and, if available, on GPU, comparing elementary against native F1–F3. This needs those frameworks installed, which was not allowed in this investigation.
4. Switch the convolution layer to F1–F3 only if the difference is significant. It is a local change: a few lines per backend and in the nerva-sympy equation blocks. The mathematics is unaffected.

## 11. Points for decision

1. **Representation:** pixel matrix `(N·H·W) × C` between convolution and pooling layers (this proposal), instead of one row per image plus `reshape` (revision 2).
2. **Names:** `im2col`/`col2im` (accepted). Block operations: `blocks_sum`, `blocks_repeat`, `blocks_max`, `blocks_max_mask`.
3. **Specification in the table:** define `im2col` and `col2im` through `S_pq` (§3–4), or through index formulas, or both. This choice affects only the documentation; §9 shows it does not affect efficiency.
4. **Data loader and training loop:** conversion to pixel matrices in the loader, and normalisation by `N` instead of `X.shape[0]` (§1).
5. **Max pooling ties:** keep the "first maximum" rule.
6. **Combined operations (§10):** whether to adopt F1–F3, and under which names (`correlate…`, `conv…`, `im2col_product…`).
