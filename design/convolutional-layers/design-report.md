# Nerva — Convolutional Layers: Architectural Investigation and Design (revision 2)

> **Revision 3 note.** The primitives and the representation in §2.5, §3 and §4 are superseded by `primitives-specification.md`. That document stores feature maps as a pixel matrix `(N·H·W) × C`, removes `reshape` and `rows_max_mask`, and defines `im2col`/`col2im` and four block operations by 0/1 selection matrices. The rest of this report (current architecture, framework survey, testing, roadmap, open questions) still applies, with those primitives swapped in.

*Scope: nerva-numpy, nerva-jax, nerva-tensorflow, nerva-torch and nerva-sympy (branch `claude/keen-wright-k5x3nq`, read-only). Date: 2026-11-01.*

**Status: proposal for maintainer review. No source code, dependencies, configuration, branches or GitHub settings have been changed.**

> **What changed since revision 1.**
> Revision 1 proposed coarse primitives (`conv2d`, `conv2d_gradient_input`, …) that would call framework-native convolution kernels and their backward passes. That breaks the Nerva rule:
>
> *every layer is expressed in named primitive operations from the matrix-operations table; only the primitives have a framework-specific implementation; the layer code is the same in every backend.*
>
> Revision 2 follows that rule. Convolution and pooling are built from **four new elementary primitives**:
> - `reshape`
> - `im2col` (a pure re-indexing)
> - `col2im` (its adjoint, which contains the summation)
> - `rows_max_mask`
>
> They are combined with **existing** primitives (`@`, `.T`, `row_repeat`, `column_repeat`, `columns_sum`, `rows_max`, `rows_mean`, `hadamard`). The weight layout also changes from (p,q,c) to (c,p,q) columns, so that conv and pooling share one `im2col`. Native convolution kernels are no longer used.

How each statement is marked:
- **[F]** fact, verified in the repositories.
- **[P]** prototype result, from throwaway NumPy code in the session scratchpad, not in any repository.
- **[E]** documented behaviour of an external framework, from my knowledge. JAX, TensorFlow and PyTorch are not installed in this environment, so **re-check these when the code is written**.
- **[R]** recommendation.
- **[A]** assumption.

---

## 0. Executive summary

1. **Layers still exchange row-layout matrices `X ∈ ℝ^{N×D}`.** For images `D = H·W·C`, flattened in NHWC (channels-last) order [R]. The image dimensions are layer metadata. [F] `prepare_data.py:170-172` already stores CIFAR-10 this way. MNIST and the shapes dataset are 28×28×1.
2. **Four new primitives are added to the matrix-operations table** [R]:

   | Code | Definition |
   |---|---|
   | `reshape(X, m, n)` | row-major reshape |
   | `im2col(X, g)` | patch matrix: a gather, no arithmetic |
   | `col2im(P, g)` | adjoint of `im2col`: a scatter-add, **this is where the summation is** |
   | `rows_max_mask(X)` | 0/1 matrix marking the first maximum of each row |

3. **The convolution layer is a linear layer on the patch matrix.** Its code is identical in every backend:
   ```
   Xc  = im2col(X, g)
   Zc  = Xc @ W.T + row_repeat(b, M)            # M = N·H′·W′
   Z   = reshape(Zc, N, H′·W′·K)
   ...
   DZc = reshape(DZ, M, K)
   DW  = DZc.T @ Xc
   Db  = columns_sum(DZc)
   DX  = col2im(DZc @ W, g)
   ```
   These are the `LinearLayer` equations with `X → im2col(X)` and `DX → col2im(·)`.
4. **Pooling uses the same `im2col`, plus a reshape**, followed by `rows_max` or `rows_mean`. Its backward pass uses `rows_max_mask`, `column_repeat`, `hadamard` and `col2im`. This works because the weight columns, and therefore the patch columns, are ordered (c, p, q).
5. **Correctness of the whole design is checked in the prototype** [P]:
   - all three layers' gradients match finite differences (≤ 2e-8) across stride, dilation, asymmetric padding, non-square kernels and overlapping pooling windows;
   - `im2col`/`col2im` are exact adjoints (≤ 1e-14);
   - convolution matches a direct loop implementation (≤ 5e-15).
6. **Each framework only implements the four primitives** (§4), e.g. PyTorch `F.unfold`/`F.fold`, JAX `conv_general_dilated_patches`, TensorFlow `extract_patches` + `unsorted_segment_sum`, and NumPy strided slices.
7. **nerva-sympy implements the same four primitives on `sp.Matrix`** using only sympy. Its symbolic tests show that `DW`, `Db` and `DX` above equal the exact gradients. The existing **textual equation checker** then covers the new layers, because their equations are identical in every backend.
8. **Training loop, data loaders, losses, optimizers, learning-rate schedulers and the `MultilayerPerceptron` chaining are unchanged.**

---

## 1. Current architecture (established facts)

### 1.1 Structure [F]
The four backends are near-identical clones (`src/nerva_<backend>/`): `matrix_operations.py`, `layers.py`, `activation_functions.py`, `softmax_functions.py`, `loss_functions.py`, `optimizers.py`, `weight_initializers.py`, `learning_rate.py`, `datasets.py`, `shapes_dataset.py`, `multilayer_perceptron.py`, `training.py`, `utilities.py`. A `diff` between backends shows mostly mechanical substitutions. There is no convolution code anywhere: `nerva/README.md:67` says "No convolutional or transformer layers (planned)".

### 1.2 The matrix-operations table is the contract [F]
`doc/asciidoc/include/matrix-operations.adoc` (maintained in `nerva/doc/shared/python` and copied by `scripts/sync_docs.py`) states: *"All equations are expressed in terms of the matrix operations in the table below."* The table lists `zeros`, `ones`, `identity`, transpose, `*`, `dot`, `hadamard`, `diag`, `Diag`, `elements_sum`, `column_repeat`, `row_repeat`, `columns_sum`, `rows_sum`, `columns_max`, `rows_max`, `columns_mean`, `rows_mean`, and the element-wise `exp`, `log`, `reciprocal`, `sqrt`, `inv_sqrt` and `log_sigmoid`. Each backend's `matrix_operations.py` implements these with that framework's own calls.

### 1.3 Layers [F]
- `Layer` (`nerva-numpy/src/nerva_numpy/layers.py:21`) has `feedforward(X) -> Y`, `backpropagate(Y, DY)` (sets `self.DX`) and `optimize(eta)`. `optimize` does nothing when `optimizer is None`.
- `LinearLayer` (`layers.py:42`): `Y = X @ W.T + row_repeat(b, N)`, with backward `DW = DY.T @ X`, `Db = columns_sum(DY)`, `DX = DY @ W`. `W` is `(K, D)`.
- `ActivationLayer` (`:91`) adds `DZ = hadamard(DY, act.gradient(Z))`. Activations only exist fused into linear layers.
- `MultilayerPerceptron.backpropagate` (`multilayer_perceptron.py:29-32`) chains layers through `layer.X`/`layer.DX` and does not depend on `D`.
- Save/load and `info()` select `isinstance(layer, LinearLayer)` and use the keys `W1, b1, …`.

### 1.4 Backend differences that matter [F]

| | NumPy | JAX | TensorFlow | PyTorch |
|---|---|---|---|---|
| Parameters | mutable, `W[:] = …` | immutable, reassigned | `tf.Variable.assign` | mutable, in place |
| Optimizer binding | `(x, Dx)` | `(obj, 'W', 'DW')` | Variables | `(x, Dx)` |
| Weight init | `set_weights_*` in place | `weights_*` return new arrays | `.assign(tf.random…)` | `W.uniform_()` … |
| Transpose in `layers.py` | `W.T` | `W.T` | `tf.transpose(W)` (normalised by the equation checker) | `W.T` |
| Autograd / jit | – | not used, no `jit` | not used | not used |

**Implication [R]:** store convolution weights as a 2-D matrix `W ∈ ℝ^{C_out × (C_in·kh·kw)}`. Then optimizers, the JAX attribute binding, TF Variables, `set_layer_weights` (`K, D = W.shape`, `weight_initializers.py:44-77`) and npz save/load all work unchanged.

### 1.5 Loss, training, data [F]
- `training.py:157` computes `DY = loss.gradient(Y, T) / X.shape[0]`, and accuracy uses `argmax(axis=1)`. Both only need an `(N, K)` network output.
- `DataLoader` slices rows of `(N, D)` arrays. No image shape is stored in the `.npz` files.

### 1.6 nerva-sympy [F]
- Depends only on `numpy` and `sympy`; the backends are optional extras.
- `tests/test_layer_*.py` contain `# feedforward` / `# backpropagation` equation blocks that are checked against `gradient(loss(Y), w)` (`src/nerva_sympy/matrix_operations.py:219`).
- `scripts/_equation_test_utils.py` compares those blocks *textually* with each backend's `layers.py`, using `EQ_VARS` (line 18) for the variable names and `LAYER_KEYS` (line 122) for the layer classes.
- Shared backend tests are generated from `templates/backend_tests` and synced by `sync_backend_tests.py`.
- The shared numerical cases in `tests/one_layer_cases` are generated by `scripts/generate_one_layer_tests.py`, which imports nerva-torch (line 31).

### 1.7 Reuse [R]

| Unchanged | Extended | New |
|---|---|---|
| `Layer` interface, `MultilayerPerceptron`, SGD loops, losses, softmax heads, optimizers, LR schedulers, `DataLoader`, all existing primitives | matrix-operations table plus `matrix_operations.py` in 5 repos (four primitives); parser and CLI (shape inference); Xavier fan-out; equation checker (`EQ_VARS`, `LAYER_KEYS`); `prepare_data.py` (store image shape) | conv and pooling layers; a geometry record; sympy tests; shared reference cases |

---

## 2. Matrix representation versus image tensors

### 2.1 Three kinds of "flattening"
1. **Flattening the dataset.** Harmless if the order is fixed. It is: NHWC.
2. **A flat matrix as the inter-layer representation, with `(H, W, C)` as metadata.** Harmless. A row-major reshape is a zero-copy view in all four frameworks, and the spatial structure can be recovered exactly from three integers.
3. **im2col.** Patches are copied into a matrix. This is now a *named primitive*, so it is part of the mathematics, not hidden in the implementation.

### 2.2 Option A: tensors between layers (NHWC or NCHW)
This would require:
- a parallel rank-4 tensor vocabulary next to the 2-D table;
- a mandatory Flatten layer;
- loaders that yield 4-D batches;
- in nerva-sympy, `sp.Array` instead of `sp.Matrix`. The `gradient`, `substitute` and `equal_matrices` helpers and the equation blocks do not work with `sp.Array`.

Whichever layout is chosen, some framework has to transpose: PyTorch is NCHW-native, TensorFlow's CPU kernels are NHWC [E]. **Rejected**: high cost, no mathematical benefit.

### 2.3 Option B: im2col / matrix convolution (chosen)
- Convolution becomes one matrix product per layer.
- Backpropagation through the transformation is its adjoint, `col2im`.
- Memory overhead is `kh·kw/(sh·sw)` times the input.
  - CIFAR network below, N = 100, float32 [P]: 10.5 MiB for conv1 against a 1.2 MiB input; 28.1 MiB for conv2 against 3.1 MiB.
  - Pooling with 2×2 windows and stride 2 has **no** overhead: the patch matrix has exactly as many elements as the input.
- Materialisation can be avoided with batch chunking, a later optimisation that does not change the layer code (§8).

### 2.4 Option C: other approaches
- **Implicit GEMM and native direct or Winograd convolution** live inside cuDNN, oneDNN or XLA kernels. Under the primitive rule they could only enter as a coarse `conv2d` primitive. Revision 1 did that, and it moves the mathematics out of the table. **Not used** [R]. A possible later opt-in fast path is §10 Q8.
- **Shift-and-accumulate**, `Z = Σ_{p,q} shift_{p,q}(X)·W_{p,q}ᵀ`, makes the summation explicit in the layer, using primitives `shift`/`unshift`.
  - It needs about 1× extra memory instead of k².
  - The layer equation would need a loop over (p,q). The equation checker only handles straight-line assignments.
  - It uses kh·kw small GEMMs, which measured 2–3× slower in NumPy [P].
  - Pooling would need an element-wise `maximum` primitive and a tie mask spread over kh·kw matrices.
  - **Second choice**, kept as an alternative (§10 Q2).

### 2.5 Recommended architecture (answers to the brief)

| Level | Representation |
|---|---|
| Dataset / DataLoader | `(N, D)` rows, NHWC order (unchanged) |
| Between layers | `(N, D)` matrices (unchanged); layers carry `in_shape`/`out_shape` |
| Parameters | `W ∈ ℝ^{C_out×(C_in·kh·kw)}` with columns (c,p,q); `b ∈ ℝ^{C_out}` |
| Layer equations | existing primitives + `reshape`, `im2col`, `col2im`, `rows_max_mask`, identical in all backends |
| Framework code | only inside the four new primitives (§4) |

- **Can Nerva keep its matrix architecture?** Yes, completely.
- **Should tensors be first-class?** No. *Shapes* become first-class: a geometry record.
- **Hybrid?** Yes in a precise sense. The 4-D view of the data exists only *inside* `im2col`/`col2im`.
- **Public API:** matrices, the geometry record, the four primitives, and the new layers.
- **Same interface efficiently in four frameworks?** Yes. Every framework has a native or near-native implementation of `im2col` and its adjoint (§4). The GEMM runs on BLAS, oneDNN, XLA or cuBLAS.

---

## 3. Definitions

### 3.1 Conventions [R]
- **Layout:** NHWC inside a row. The flat index is `(h·W + w)·C + c`.
- **Operation:** cross-correlation (no kernel flip), which is what all frameworks call convolution [E].
- **Geometry:** `g = Conv2DGeometry(H, W, C, kh, kw, sh=1, sw=1, dh=1, dw=1, pt=0, pb=0, pl=0, pr=0)`. This is a plain Python record, identical in every backend (no framework code). It provides
  `H′ = ⌊(H+pt+pb−dh(kh−1)−1)/sh⌋+1` and `W′ = ⌊(W+pl+pr−dw(kw−1)−1)/sw⌋+1`.
  The constructor rejects `H′ < 1` and non-positive parameters.
- **Padding:** explicit `(pt,pb,pl,pr)` with zeros. The parser converts `valid` and `same` into explicit values; `same` uses TensorFlow's rule [E]: `total = max((⌈H/s⌉−1)s + dh(kh−1)+1−H, 0)`, `pt = ⌊total/2⌋`, `pb = total − pt`.
- **Rounding:** floor only.
- **Dtype:** primitives preserve the input dtype.

### 3.2 New primitives (proposed rows for the matrix-operations table)

| Operation | Code | Definition |
|---|---|---|
| reshape | `reshape(X, m, n)` | the m×n matrix with the elements of `X` in row-major order (`m·n = size(X)`) |
| patch matrix | `im2col(X, g)` | `X ∈ ℝ^{N×HWC}` ↦ `P ∈ ℝ^{(N·H′·W′)×(C·kh·kw)}`, with `P[(n,i,j),(c,p,q)] = X̃[n, sh·i+dh·p−pt, sw·j+dw·q−pl, c]`, where `X̃` is 0 outside the image. Rows are ordered (n,i,j); columns (c,p,q) |
| patch sum | `col2im(P, g)` | `ℝ^{(N·H′·W′)×(C·kh·kw)} → ℝ^{N×HWC}`: `col2im(P)[n,(h,w,c)] = Σ_{i,p: sh·i+dh·p−pt=h} Σ_{j,q: sw·j+dw·q−pl=w} P[(n,i,j),(c,p,q)]` |
| row-max indicator | `rows_max_mask(X)` | `m×n` 0/1 matrix with a 1 at the **first** maximal element of each row |

**Properties [P]:**
- `im2col` is linear and `col2im` is its adjoint: `⟨im2col(X), P⟩ = ⟨X, col2im(P)⟩`. Equivalently, if `vec(im2col X) = S·vec(X)` for a 0/1 selection matrix `S`, then `vec(col2im P) = Sᵀ·vec(P)`. This is the formal matrix definition, and it lets the backward equation be derived like every other Nerva layer.
- The summation in `col2im` runs over all patches that contain a given pixel. Overlapping patches (stride < effective kernel) accumulate. Padded positions are dropped. Pixels never covered (when the stride does not divide the extent evenly) get 0.
- `rows_max_mask` is defined without `argmax`, as `E = (X == column_repeat(rows_max(X), n))` followed by "keep the first 1 in each row" (a running sum equal to 1). This makes tie handling deterministic and identical in every framework. Note that TensorFlow does not guarantee which index `tf.argmax` returns on ties [E].

**Why the weight columns are ordered (c,p,q).** Row-major reshaping `im2col(X) ∈ ℝ^{(N·H′·W′)×(C·kh·kw)}` to `ℝ^{(N·H′·W′·C)×(kh·kw)}` gives one row per (n,i,j,c) and one column per window offset (p,q). That is exactly the matrix that pooling needs. With (p,q,c) order, pooling would need an extra transpose primitive. As a side effect, `W.reshape(C_out, C_in, kh, kw)` is PyTorch's weight layout [E].

### 3.3 Convolution layer

Shapes: `X (N, H·W·C)`, `W (K, C·kh·kw)`, `b (K,)`, with `M = N·H′·W′`.
```
# feedforward
Xc = im2col(X, g)
Zc = Xc @ W.T + row_repeat(b, M)
Z  = reshape(Zc, N, H′·W′·K)        # (N,i,j)×k row-major == NHWC-flat: no transpose needed
Y  = act(Z)                         # Conv2DActivationLayer; plain Conv2DLayer has Y = Z

# backpropagation
DZ  = hadamard(DY, act.gradient(Z))
DZc = reshape(DZ, M, K)
DW  = DZc.T @ Xc
Db  = columns_sum(DZc)
DX  = col2im(DZc @ W, g)
```
Index form, for reference:
- `Z[n,i,j,k] = b_k + Σ_{c,p,q} W[k,(c,p,q)]·X̃[n,sh·i+dh·p−pt,sw·j+dw·q−pl,c]`
- `DW[k,(c,p,q)] = Σ_{n,i,j} DZ[n,i,j,k]·X̃[…]`
- `Db_k = Σ_{n,i,j} DZ[n,i,j,k]`
- `DX = col2im` of `Σ_k DZ[n,i,j,k]·W[k,(c,p,q)]`

The summations over n,i,j and k sit inside the matrix products; the summation over (i,p),(j,q) sits in `col2im`.

Storing `Xc` in `feedforward` mirrors how `LinearLayer` keeps `X`. The alternative, recomputing it in `backpropagate`, is a memory/time trade-off, listed as §10 Q6.

### 3.4 Max pooling layer
Geometry with kernel `(kh,kw)` and stride defaulting to the kernel; no padding in v1. Zero padding would be wrong for max pooling. Here `M = N·H′·W′·C`.
```
# feedforward
P = reshape(im2col(X, g), M, kh·kw)
Y = reshape(rows_max(P), N, H′·W′·C)

# backpropagation
DP = hadamard(rows_max_mask(P), column_repeat(reshape(DY, M, 1), kh·kw))
DX = col2im(reshape(DP, N·H′·W′, C·kh·kw), g)
```
- The gradient goes to the first maximum of each window, and overlapping windows accumulate in `col2im`.
- Ties after ReLU (all-zero windows) do not matter in training: those positions have `relu′ = 0` downstream. They do matter for bit-exact tests of the layer itself, which the deterministic mask handles.

### 3.5 Average pooling layer
```
# feedforward
P = reshape(im2col(X, g), M, kh·kw)
Y = reshape(rows_mean(P), N, H′·W′·C)

# backpropagation
DP = column_repeat(reshape(DY, M, 1), kh·kw) / (kh·kw)
DX = col2im(reshape(DP, N·H′·W′, C·kh·kw), g)
```
Global average pooling is the case `kh = H`, `kw = W`.

### 3.6 Flattening
Flattening is the identity: the output of a conv or pooling layer already is `(N, H′·W′·C)`, NHWC-flat. No Flatten layer is needed. A dense layer that follows simply has `D = H′·W′·C`.

### 3.7 Operations versus layers
- The four primitives go into `matrix_operations.py`. They are elementary and reusable; for example, `rows_max_mask` and `reshape` will be useful elsewhere.
- The new layers are `Conv2DLayer`, `Conv2DActivationLayer`, `MaxPool2DLayer` and `AvgPool2DLayer`. The pooling layers have no parameters and `optimizer = None`.
- No "pooling operation" is needed beyond `rows_max` and `rows_mean`, which are already in the table.

---

## 4. Framework implementations of the four primitives

Only these functions differ between backends. [E]: verify each API at implementation time.

| Primitive | NumPy | JAX | TensorFlow | PyTorch |
|---|---|---|---|---|
| `reshape` | `X.reshape(m, n)` | `jnp.reshape` | `tf.reshape` | `X.reshape(m, n)` |
| `im2col` | `np.pad`, then kh·kw strided-slice copies into an `(N,H′,W′,C,kh,kw)` buffer, then reshape (prototype) | `lax.conv_general_dilated_patches(X4, (kh,kw), (sh,sw), [(pt,pb),(pl,pr)], rhs_dilation=(dh,dw), dimension_numbers=('NHWC','OIHW','NHWC'))`; its feature dimension is documented as channel-major (c,p,q), so check the order with a test. Fallback: the NumPy slice algorithm with `jnp.stack` | `tf.image.extract_patches(tf.pad(X4,…), sizes=[1,kh,kw,1], strides=[1,sh,sw,1], rates=[1,dh,dw,1], padding='VALID')` gives (p,q,c) order, then reshape and transpose to (c,p,q). Fallback: slices plus `tf.stack` | `F.unfold(F.pad(X4.permute(0,3,1,2), (pl,pr,pt,pb)), (kh,kw), dilation, 0, stride)` gives `(N, C·kh·kw, L)` in (c,p,q) order, then `.transpose(1,2).reshape(M, -1)` |
| `col2im` | zero buffer, kh·kw strided-slice `+=`, crop the padding | `Xp.at[:, sl_h, sl_w, :].add(…)` for each (p,q), then crop; or `jax.linear_transpose(im2col)` | `tf.math.unsorted_segment_sum` with a per-geometry index table (each patch entry → its padded pixel), then crop. Alternative: `tf.tensor_scatter_nd_add` | `F.fold(P.reshape(N,L,-1).transpose(1,2), (H+pt+pb, W+pl+pr), (kh,kw), dilation, 0, stride)`, then crop. `fold` is the sum-adjoint of `unfold` |
| `rows_max_mask` | `E = X == X.max(1, keepdims=True)`; `E & (cumsum(E,1) == 1)` | same with `jnp` | same with `tf.math.cumsum` | same with `torch.cumsum` |

**Uniform fallback for all four:** precompute, once per geometry, an integer index table that maps each patch entry to its padded pixel. Then `im2col` is a gather (`take`, `jnp.take`, `tf.gather`, `torch.index_select`), and `col2im` is a segment-sum or scatter-add (`np.add.at` or `np.bincount`, `jax.ops.segment_sum`, `tf.math.unsorted_segment_sum`, `torch.index_add_`). The index table can be built by shared, framework-free code.

**Consequences:**
- None of the backends uses autograd or native convolution kernels any more. Backpropagation is explicit everywhere, as for the existing layers.
- In JAX and TF the primitives are functional. In TF and torch they work with Variables and tensors as they are. Nothing interacts with the parameter-storage differences of §1.4, because the primitives only touch activations.
- The GPU precision caveats now apply only to the GEMM, as they already do for MLPs: TF32 matmul is default in TF on Ampere, and JAX's default precision applies [E]. Integer cases on CPU stay bit-exact.

---

## 5. Symbolic validation with nerva-sympy

### 5.1 Additions (only sympy and numpy) [R]
1. **`src/nerva_sympy/matrix_operations.py`:** `reshape` (sympy's `Matrix.reshape` is row-major), `im2col`, `col2im` and `rows_max_mask`, written as index loops over `sp.Matrix` entries. The geometry record goes in a small shared module.
2. **`tests/test_layer_convolution.py`.** This is the `test_layer_linear.py` pattern with `X` replaced by `Xc`:
   - `x = matrix('x', N, H*W*C)`, `w = matrix('w', K, C*kh*kw)`, `b = matrix('b', 1, K)`;
   - the equation block of §3.3, verbatim;
   - assert `DW == gradient(loss(Y), w)`, `Db == gradient(loss(Y), b)` and `DX == gradient(loss(Y), x)` for `loss ∈ {elements_sum, squared_error}` and several activations.

   This **proves** that `col2im` is the correct backward operator, rather than just testing it. Suggested sizes are N = 2, H = W = 4, C = 2, K = 2, k = 3, plus stride 2, dilation 2 and asymmetric-padding variants marked `@slow`.
3. **`tests/test_convolution_operations.py`:**
   - `im2col` against its index definition;
   - adjointness `elements_sum(hadamard(im2col(X), P)) == elements_sum(hadamard(X, col2im(P)))` with symbolic `X` and `P`;
   - `reshape` round trips;
   - `rows_max_mask` on concrete rationals, including ties.
4. **`tests/test_layer_pooling.py`:**
   - Average pooling: fully symbolic (it is linear).
   - Max pooling: build `Y` with `rows_max` (`sp.Max`) on symbols, take `gradient(loss(Y), x)`, then substitute **distinct rationals** and compare with the §3.4 `DX`. This is exact, because the result is piecewise.
5. **Equation checker:**
   - add `Conv2DLayer`, `Conv2DActivationLayer`, `MaxPool2DLayer` and `AvgPool2DLayer` to `LAYER_KEYS`;
   - add `Xc`, `Zc`, `DZc`, `P` and `DP` to `EQ_VARS`.

   Because the layer code is identical in all backends, this check now covers convolution in exactly the same way it covers `LinearLayer`.

### 5.2 Shared reference cases without framework dependencies [R]
- A new generator, `scripts/generate_conv_cases.py`, depends on **numpy only** (unlike `generate_one_layer_tests.py`, which imports nerva-torch).
- It writes `tests/conv_cases/*.npz` plus `manifest.json` and copies them to the siblings, like `one_layer_cases`. Each case holds:
  - the geometry;
  - `X`, `W`, `b`, `DY`;
  - expected outputs for each primitive (`im2col`, `col2im`, `rows_max_mask`);
  - expected outputs for each layer (`Z`/`Y`, `DX`, `DW`, `Db`).
- **Exact cases:** integers in [−3, 3], 2×2 average pooling (division by 4 is exact). float32 is then bit-identical to float64 [P], so the tolerance is 0.
- **Random cases:** checked with the tolerances in §7.
- **Templates:** `templates/backend_tests/test_convolution_operations.py` and `test_convolution_layers.py`, synced to the backends.
- **Cross-framework (on request):** primitives in all four backends on the same inputs, plus a short CNN training trace on the 28×28×1 shapes dataset (no download).

---

## 6. CIFAR-10 example

Network: 32×32×3 → Conv 3×3/32, padding 1, ReLU → MaxPool 2/2 → Conv 3×3/64, padding 1, ReLU → MaxPool 2/2 → Linear 4096→10 (softmax via the `SoftmaxCrossEntropy` loss, as in the existing CIFAR script).

| Stage | H×W×C | D | Primitive sizes (N = 100) | Parameters | FLOP/sample (fwd) |
|---|---|---|---|---|---|
| Input | 32×32×3 | 3,072 | – | – | – |
| Conv1 + ReLU | 32×32×32 | 32,768 | `Xc` 102,400×27 (10.5 MiB) | 32·27+32 = **896** | 1.77 M |
| MaxPool | 16×16×32 | 8,192 | `P` 819,200×4 (12.5 MiB, = input size) | 0 | – |
| Conv2 + ReLU | 16×16×64 | 16,384 | `Xc` 25,600×288 (28.1 MiB) | 64·288+64 = **18,496** | 9.44 M |
| MaxPool | 8×8×64 | 4,096 | `P` 409,600×4 (6.3 MiB) | 0 | – |
| Linear | 10 | 10 | – | 4096·10+10 = **40,970** | 0.08 M |
| **Total** | | | | **60,362** | ≈ 11.3 M fwd, ≈ 34 M per training step |

- **Memory, N = 100, float32:** activations ≈ 43 MiB, plus stored `Xc` and `P` ≈ 57 MiB, plus backward transients ≈ 40 MiB. Peak ≈ **140 MiB**, about 1.4 GB at N = 1000.
- **NumPy time, prototype, not optimised, 4 cores [P]:**
  - conv1: forward 70 ms, backward 17 ms per batch of 100;
  - conv2: forward 45 ms, backward 22 ms;
  - max pooling on 32×32×32: about 160 ms for forward plus backward.

  That is roughly 1–3 minutes per epoch. The slow parts are the (c,p,q)-ordered `im2col` copy and the cumsum-based mask, both of which have obvious optimisations inside the primitives (§8).
- **Consistency:** every operation is a table primitive, so the network is the same code in all four backends.

---

## 7. Testing strategy

1. **Shapes:** a parametrised grid over H, W ∈ {1,4,5,7}, k ∈ {1,2,3}, s ∈ {1,2,3}, d ∈ {1,2}, symmetric and asymmetric padding, C, K ∈ {1,3}. Check `H′, W′`, the sizes of the primitive outputs, and that invalid geometries are rejected.
2. **Primitive tests against `conv_cases`.** Exact cases use tolerance 0; random cases use the §7 rule below. Cover in particular:
   - padding cuts, and uncovered trailing rows (`col2im` gives 0 there);
   - overlap accumulation;
   - 1×1 kernels, kernel equal to the input, N = 1;
   - `rows_max_mask` with ties.
3. **Adjointness:** `⟨im2col X, P⟩ = ⟨X, col2im P⟩`. A single scalar check that catches index errors.
4. **Layer tests** (mirroring `test_one_layer.py`): `Z`/`Y`, `DX`, `DW` and `Db` against the cases, plus one optimizer step.
5. **Finite differences:** float64 only, on layers and primitives.
   - `h = ε^{1/3}·max(1,|x|) ≈ 6e-6`; expected error ≈ 1e-9 to 1e-8 [P].
   - Directional checks along random `v` are cheapest.
   - Use tie-free inputs and keep away from ReLU kinks.
   - Float32 finite differences (error ≈ 1e-4) are not a meaningful check.
6. **Equation checker** (`test_equations.py`).
7. **Cross-backend** agreement of the primitives and a short CNN training trace.
8. **Existing MLP tests unchanged.**

**Tolerances:**
- **Bound per element:** `|a−b| ≤ F·u·B`, where:
  - `u = 2⁻²⁴` (float32) or `2⁻⁵³` (float64);
  - `B` is the same computation on absolute values;
  - `F` is the reduction length:

    | Quantity | F |
    |---|---|
    | `Z` | `C·kh·kw` |
    | `DX` | `K·kh·kw` |
    | `DW` | `N·H′·W′` |
    | `Db` | `N·H′·W′` |

  [P] Measured float32 error was 0.007 of this bound for conv2 at CIFAR size (relative error 5e-7).
- **Simpler rule:** float32 `rtol = atol/max|ref| = 1e-5`, raised to `1e-4` for `DW`/`Db` with long reductions; float64 `1e-12`.
- **Primitive-only operations:** `im2col`, `reshape` and `rows_max_mask` must be **exact** in every dtype. `col2im` sums at most `⌈kh·dh/sh⌉·⌈kw·dw/sw⌉` terms; integer cases are exact.

---

## 8. Performance and scalability

| | Cost | Extra memory | CPU | GPU |
|---|---|---|---|---|
| im2col + GEMM (chosen) | GEMM ≈ 2·M·K·C·kh·kw; im2col/col2im are memory-bound | `kh·kw/(sh·sw)` × input | GEMM via BLAS/oneDNN; NumPy im2col via kh·kw slices; torch `unfold`/`fold` are native C++ | `unfold` + cuBLAS: typically a few times slower than cuDNN and uses k² memory [A] |
| Native kernels (revision 1) | same FLOPs, often less (Winograd) | workspace only | best | best |
| Shift-and-accumulate | same FLOPs | ≈ 1× | 2–3× slower in NumPy at CIFAR sizes [P] | many small GEMMs |

- **Optimisations that keep the layer code unchanged** (they change only primitive implementations):
  - build `im2col` in (p,q,c) order and transpose once, for cache-friendly writes;
  - replace the cumsum mask with an argmax one-hot where the framework guarantees first-index ties (NumPy, PyTorch, JAX [E]);
  - batch chunking for large feature maps. This needs a matching chunked GEMM, so it is better as a later, layer-internal option;
  - skip `DX` in the first layer.
- **Gap:** under the primitive rule, the GPU backends give up cuDNN's implicit-GEMM and Winograd speed. This is the main trade-off of the rule. It is acceptable for Nerva's goals of transparency and verifiability [A]. A separate opt-in fast path would break code identity and is left to the maintainer (§10 Q8).

---

## 9. Implementation roadmap

| # | Milestone | Repos / main files | Depends | Deliverables | Tests | Risks / decisions |
|---|---|---|---|---|---|---|
| M1 | Conventions and table rows | `nerva/doc/shared/python` (matrix-operations table, synced to the backends) | – | NHWC; (c,p,q) weights; geometry record; the definitions of the 4 primitives; tie rule; `same`/`valid` | – | §10 decisions |
| M2 | Sympy primitives + proofs | `nerva-sympy/src/nerva_sympy/matrix_operations.py`, `tests/test_convolution_operations.py`, `test_layer_convolution.py`, `test_layer_pooling.py`, `scripts/_equation_test_utils.py` (`EQ_VARS`, `LAYER_KEYS`) | M1 | symbolic proofs of §3.3–3.5 | symbolic (some `@slow`) | sympy runtime: keep sizes tiny |
| M3 | Reference cases | `nerva-sympy/scripts/generate_conv_cases.py`, `tests/conv_cases/`, `templates/backend_tests/test_convolution_*.py` | M2 | exact and random cases | generator checked against sympy | stable, versioned manifest |
| M4 | NumPy primitives | `nerva-numpy/.../matrix_operations.py` (+ geometry module) | M3 | 4 primitives | `conv_cases` primitive tests, adjointness | – |
| M5 | NumPy layers + construction | `layers.py`, `multilayer_perceptron.py` (shape-inferring parser), `weight_initializers.py`, `tools/mlp.py` or `tools/cnn.py` | M4 | 4 layers; spec syntax such as `Conv2D(channels=32,kernel=3,padding=same,act=ReLU);MaxPool2D(2);…;Linear` with `--input-shape=32,32,3` | layer tests, float64 finite differences, save/load, equation checker | inheritance versus mixin; CLI compatibility |
| M6 | JAX, TF and PyTorch primitives, then the layers *copied verbatim* (3 independent PRs) | `matrix_operations.py`, `layers.py`, parser, initializers | M5 | §4 implementations | same synced templates plus equation checker | column order of `conv_general_dilated_patches` / `extract_patches`; TF `col2im` via segment sum; `F.unfold` padding |
| M7 | Cross-framework | `nerva-sympy/tests/cross_framework/` | M6 | primitive agreement plus a CNN trace on the shapes dataset | on request | – |
| M8 | Examples, CIFAR | `prepare_data.py` (optional `image_shape` key), CNN example scripts, docs | M5/M6 | runnable examples (shapes dataset by default) | `run_all_examples.py` smoke test | keep CI download-free |
| M9 | Optimisation where measured | primitive implementations only | M8 | §8 items | same suites must pass unchanged | none to layer code |
| later | Spatial batch normalisation | all | M6 | BN on `reshape(Z, N·H·W, C)`: the existing BN equations apply as they are | sympy plus cases | the existing BN on a flat conv output normalises per pixel and channel, which is not standard spatial BN; document this |

---

## 10. Questions for the maintainer

1. **Primitive set and names.**
   - Confirm `reshape`, `im2col`, `col2im` and `rows_max_mask`.
   - Alternative name pairs: `patches` / `patches_adjoint`, or `unfold` / `fold`.
   - Should `col2im` be written in the table as `im2colᵀ` (the adjoint, defined via the selection matrix `S`) or by its summation formula? Or both?
2. **Granularity.** im2col/col2im (recommended), or shift/unshift per kernel offset? The latter makes the summation over (p,q) explicit in the layer, but needs a loop in the equations.
3. **Geometry argument.** Should the primitives take a geometry record `g` (recommended, short and checkable), or explicit integers like the existing primitives?
4. **Layout.** NHWC data with (c,p,q) weight columns (recommended), or (p,q,c) weights plus an extra transpose primitive for pooling?
5. **Xavier fan-out.** Keep `W.shape[0] = C_out`, or use `C_out·kh·kw` (PyTorch/Keras)? He initialisation is already correct.
6. **Store or recompute `Xc` / `P`** in `feedforward`: memory against time.
7. **Class hierarchy and activations.** Should `Conv2DLayer` subclass `LinearLayer`? Fused `Conv2DActivationLayer`, or a standalone activation layer?
8. **Native fast path.** Should an optional coarse `conv2d` primitive ever be allowed for GPU speed, or is code identity absolute?
9. **Pooling scope for v1:** no padding, no `ceil_mode`?
10. **Model construction and metadata:** a new shape-inferring parser, `--input-shape`, an `image_shape` key in the npz files?
11. **Dtype policy:** dtype-generic primitives; float64 at layer level?
12. **Generators:** should the nerva-torch dependency of `generate_one_layer_tests.py` also be removed later?

---

### Appendix: prototype evidence (scratchpad, not committed)
`proto2.py` (NumPy 2.5.3, Python 3.13, 4 CPUs) implements the four primitives and the three layers exactly as in §3. All layer code uses only primitives.
- **Conv forward** matches a direct loop implementation to ≤ 5e-15 in 5 geometries: stride (2,1), (2,2) and (3,3); dilation 2; asymmetric padding; non-square kernels.
- **Conv finite differences:** `DX`, `DW` and `Db` agree to ≤ 2e-8.
- **Adjointness** holds to ≤ 1e-14.
- **Pooling finite differences:** max and average pooling with (k,s) = (2,2), (3,2) and (3,1) agree to ≤ 4e-9.
- **`rows_max_mask` with ties** gives `[1,3,3,0] → [0,1,0,0]`, `[2,2,2,2] → [1,0,0,0]` and `[0,0,0,0] → [1,0,0,0]`.
- **Integer inputs** give float32 results bit-identical to float64 (`proto.py`).
- **Timings:** see §6.
