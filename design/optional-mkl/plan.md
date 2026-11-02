# Plan: making Intel MKL an optional dependency of nerva-rowwise and nerva-colwise

Status: proposal (not implemented). Applies equally to `nerva-rowwise` and
`nerva-colwise`; both repositories have the same MKL footprint (the files
below exist in both, with layout-specific differences only).

## 1. Goals and constraints

1. A build without MKL must be possible on any platform with a C++17 compiler
   (Linux/macOS/Windows, x86-64 and ARM64), using only Eigen plus, optionally,
   a generic BLAS.
2. **No runtime penalty for the current setup.** When MKL is enabled, the
   generated code for the hot paths must be identical to today's: the same MKL
   calls, no extra indirection (no virtual dispatch, no function pointers, no
   runtime branch per operation beyond the ones that already exist), and no
   extra copies.
3. Eigen stays a hard dependency. It is also the natural fallback for all
   matrix operations.
4. Keep the public API (`sparse_linear_layer`, the `--computation` option,
   Python bindings, saved weight files) source- and file-compatible.

The approach is therefore **compile-time backend selection**: a CMake option
picks the backend, the corresponding headers are included, and all dispatch is
resolved by the preprocessor / `if constexpr` / templates.

## 2. Inventory of the current MKL dependency

### 2.1 Build system

| Location | MKL usage |
|---|---|
| `CMakeLists.txt` | `find_package(MKL CONFIG REQUIRED)`; `EIGEN_USE_MKL_ALL`, `MKL_ILP64`; LAPACKE workaround define `EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H`; RPATH set to `${MKL_ROOT}/lib`; `MKL::MKL` linked into `nervalib` and the pybind11 module |
| `tests/CMakeLists.txt`, `tools/CMakeLists.txt`, `examples/CMakeLists.txt` | link `MKL::MKL`; build `mkl_test`, `mkl_matrix_test`, `sycl_test`, `tools/mkl.cpp`, `examples/subnormal_numbers.cpp` |
| `Jamroot`, `tools/Jamfile` | hard-coded `$(ONEAPI_ROOT)/mkl/latest`, MKL link groups |
| `python/setup.py` | raises if `ONEAPI_ROOT`/`MKL_ROOT` is not set; links static MKL libs |

### 2.2 Headers that include `<mkl.h>` (all in `include/nerva/neural_networks/`)

| Header | MKL functionality used | Hot path? |
|---|---|---|
| `mkl_sparse_matrix.h` | `sparse_matrix_csr<T>` stores CSR arrays (`std::vector<MKL_INT>`) **and** an MKL handle `sparse_matrix_t`, `matrix_descr`; `mkl_sparse_?_create_csr`, `mkl_sparse_destroy`, `mkl_sparse_?_mm` (in `dsd_product`/`dds_product`); `MKL_INT` in `csr_matrix_builder` | **Yes** – the sparse feedforward and `DX` backprop of every sparse layer |
| `mkl_dense_matrix.h` | `cblas_?gemm` in `ddd_product` (the `mkl`/`blas` computation modes and `sdd_product_batch`); `mkl_?omatcopy`, `mkl_?imatcopy` (layout change / in-place transpose); `CBLAS_LAYOUT`, `CBLAS_TRANSPOSE` | **Yes** – `sdd_product_batch` (the `DW` backprop of sparse layers) and the non-Eigen dense modes |
| `mkl_dense_vector.h` | Level-1 CBLAS (`axpy`, `scal`, `dot`, `nrm2`, `asum`, `copy`, `swap`, `i?amax`, `i?amin`); MKL VML (`v?AddI`, `v?MulI`, `v?SqrtI`, `v?LinearFracI`, ...); optional SYCL code | Level-1 BLAS: optimizers in `mkl`/`blas` mode. VML: tests only |
| `mkl_eigen.h` | glue between Eigen and the above (`make_dense_matrix_view`, `to_eigen`, `to_csr`, `ddd_product`, `sdd_product*`) | via the above |

### 2.3 Indirect users of the `mkl::` namespace

`layers.h`, `optimizers.h`, `prune.h`, `grow.h`, `grow_dense.h`,
`prune_dense.h`, `regrow.h`, `weights.h`, `masking.h`, `layer_algorithms.h`,
`check_gradients.h`, `mlp_algorithms.h` (also `py::array_t<MKL_INT>` when
saving sparse weights), `multilayer_perceptron.h`, `src/python-bindings.cpp`
(`m.def("set_num_threads", mkl_set_num_threads)`).

Almost all of these only use `mkl::sparse_matrix_csr<T>` as a *container*
(traversal, pruning, growing, conversion to/from Eigen). Only a handful of
functions actually call into MKL.

### 2.4 Implicit dependency through Eigen

`EIGEN_USE_MKL_ALL` makes Eigen itself forward to MKL:

* dense `gemm`/`gemv` (`EIGEN_USE_BLAS`) – this is what the default
  `computation::eigen` path of dense layers (`X * W.transpose()`) actually
  runs on today;
* vectorised math on arrays (`EIGEN_USE_MKL_VML`: `exp`, `log`, `tanh`, ...);
* LAPACKE (`EIGEN_USE_LAPACKE`) – not relevant for nerva, and the source of
  the `EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H` workaround.

Without MKL these silently fall back to Eigen's own kernels, which is fine
functionally but must be taken into account in the benchmarks (section 6).

### 2.5 The expensive operations

For an MLP training step the cost is dominated by three kernels:

| Name in code | Operation | Used by | Current implementation |
|---|---|---|---|
| `ddd_product` | dense = dense · dense (GEMM) | dense layers (fwd + both bwd products); `sdd_product_batch` internally | Eigen `operator*` → MKL `?gemm` (via `EIGEN_USE_BLAS`), or direct `cblas_?gemm` |
| `dds_product` / `dsd_product` | dense = dense · sparseᵀ (SpMM) | sparse layers: feedforward and `DX` | `mkl_sparse_?_mm` |
| `sdd_product_batch` | sparse ⊙= dense · dense, only on the sparsity pattern (SDDMM) | sparse layers: `DW` | batched `cblas_?gemm` + gather into CSR values |

Everything else (activations, softmax, loss, optimizer updates) is
memory-bound element-wise work for which Eigen is already competitive.

## 3. Design

### 3.1 Build configuration

Add a single CMake cache variable:

```cmake
set(NERVA_BACKEND "auto" CACHE STRING "Linear algebra backend: auto, mkl, blas, eigen")
set_property(CACHE NERVA_BACKEND PROPERTY STRINGS auto mkl blas eigen)
```

* `auto` (default): `find_package(MKL CONFIG QUIET)`; if found use `mkl`,
  else `find_package(BLAS)` → `blas`, else `eigen`. Existing users who have
  oneAPI installed therefore get exactly today's build without changing their
  command line.
* `mkl`: today's configuration, `find_package(MKL CONFIG REQUIRED)`, defines
  `NERVA_USE_MKL`, `EIGEN_USE_MKL_ALL`, `MKL_ILP64`, the LAPACKE workaround and
  the MKL RPATH.
* `blas`: any CBLAS (OpenBLAS, BLIS/AOCL-BLIS, Apple Accelerate, Arm
  Performance Libraries); defines `NERVA_USE_BLAS` and `EIGEN_USE_BLAS`
  (`EIGEN_USE_BLAS` only needs the Fortran BLAS symbols, which all of these
  provide).
* `eigen`: no external library; defines nothing.

The selected backend is recorded in a generated header
`nerva/neural_networks/config.h` (via `configure_file`) so that installed
headers and the Python extension agree on it. All MKL-specific
definitions move from `add_compile_definitions` to
`target_compile_definitions(nervalib PUBLIC ...)` so they propagate to tests,
tools and examples through `nervalib` instead of being global.

The same three-way switch is added to `python/setup.py` (environment variable
`NERVA_BACKEND`, defaulting to `mkl` if `ONEAPI_ROOT`/`MKL_ROOT` is set, else
`eigen` – no more `RuntimeError`) and to the `Jamroot` (a `<nerva-backend>`
feature; lowest priority, it could also simply be documented as MKL-only).

### 3.2 Header reorganisation

Split each `mkl_*.h` header into a backend-neutral part and backend
implementations. The neutral part contains the data structures and every
algorithm that does not need an external library.

```
include/nerva/neural_networks/
  config.h                       (generated: NERVA_USE_MKL / NERVA_USE_BLAS)
  linalg/index.h                 nerva::linalg::index_type
  linalg/dense_matrix.h          dense_matrix_view, dense_matrix, submatrix view (from mkl_dense_matrix.h)
  linalg/dense_vector.h          dense_vector_view (from mkl_dense_vector.h)
  linalg/sparse_matrix_csr.h     CSR container, builder, traversal, pruning helpers
  linalg/eigen_glue.h            make_dense_matrix_view, to_eigen, to_csr (from mkl_eigen.h)
  linalg/products.h              ddd_product, dds_product, dsd_product, sdd_product[_batch]
                                 -> #includes exactly one of:
  linalg/backend_mkl.h           today's MKL code, moved verbatim
  linalg/backend_blas.h          CBLAS gemm + Eigen sparse
  linalg/backend_eigen.h         Eigen only
  linalg/vml_mkl.h               VML wrappers (only built with MKL; tests only)
  mkl_sparse_matrix.h, mkl_dense_matrix.h, mkl_dense_vector.h, mkl_eigen.h
                                 thin compatibility headers that include the above
```

The namespace `nerva::mkl` is kept as an alias of the new namespace
(`namespace mkl = linalg;`) so that `mkl::sparse_matrix_csr`,
`mkl::dds_product`, etc. in `layers.h` and friends continue to compile
unchanged. Renaming the call sites can be done afterwards as a purely
mechanical change; it is not needed for this plan.

### 3.3 The sparse matrix type

This is the only structural change. Today `sparse_matrix_csr<T>` owns the CSR
arrays *and* an MKL handle. The new class keeps the arrays and makes the
handle a backend-specific, zero-size-when-unused member:

```cpp
namespace nerva::linalg {

#if defined(NERVA_USE_MKL)
using index_type = MKL_INT;           // long long under MKL_ILP64: unchanged
#else
using index_type = std::int64_t;      // same width -> same .npy files
#endif

template <typename T>
class sparse_matrix_csr
{
    long m_rows, m_columns;
    std::vector<index_type> m_row_index;
    std::vector<index_type> m_col_index;
    std::vector<T> m_values;
    backend::csr_handle<T> m_handle;   // MKL: owns sparse_matrix_t + matrix_descr
                                       // others: empty struct, all members no-ops
  public:
    void construct_csr() { m_handle.rebuild(*this); }  // same call sites as today
    ...
};

}
```

* With MKL, `backend::csr_handle<T>` contains exactly today's
  `sparse_matrix_t m_csr` and `matrix_descr m_descr`, and `rebuild` /
  destructor / copy behaviour is today's `construct_csr` / `destruct_csr`
  code, moved. Object layout and generated code are unchanged.
* Without MKL it is an empty struct with `[[no_unique_address]]` (C++20) or
  empty-base optimisation (C++17), and `rebuild` is an inline no-op. The CSR
  arrays are viewed by the backend on demand with
  `Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor, index_type>>`
  (zero copy; Eigen accepts external compressed storage).
* Keeping the index type at 64 bits in all backends preserves the binary
  format written by `mlp_algorithms.h` (`py::array_t<MKL_INT>` becomes
  `py::array_t<linalg::index_type>`), so weight files remain interchangeable
  between MKL and non-MKL builds. A 32-bit index is a possible later
  optimisation (smaller memory traffic in SpMM), but only behind an explicit
  option and with a conversion on load/save.

### 3.4 Backend implementations of the kernels

| Kernel | `mkl` | `blas` | `eigen` |
|---|---|---|---|
| `ddd_product` | `cblas_?gemm` (unchanged) | `cblas_?gemm` (same code; `CBLAS_LAYOUT` etc. come from `cblas.h`) | `C.noalias() = A * B` on `Eigen::Map`s of the views (respects transposes and layouts) |
| `dds_product` / `dsd_product` | `mkl_sparse_?_mm` (unchanged) | Eigen: `C.noalias() = alpha * A * Wmapᵀ + beta * C` with a row-major `Map` of the CSR (OpenMP-parallel in Eigen for row-major sparse × dense) | same as `blas` |
| `sdd_product_batch` | batched `cblas_?gemm` + gather (unchanged) | same algorithm (portable already, only needs `ddd_product`) | same algorithm with the Eigen `ddd_product` |
| matrix copy / transpose (`mkl_?omatcopy`, `mkl_?imatcopy`) | unchanged | Eigen assignment (`B = alpha * A.transpose()`) on `Map`s; in-place transpose via a temporary (the code itself notes that `mkl_?imatcopy` is slow) | same |
| Level-1 BLAS (`axpy`, `scal`, ...) in `optimizers.h` | unchanged | `cblas_?axpy` etc. | Eigen expressions on `Map`s (`y += a * x`) |
| `i?amin` | MKL | OpenBLAS/BLIS provide it; otherwise Eigen `minCoeff` | Eigen |
| VML (`v?AddI`, ...) | unchanged | not provided | not provided |
| SYCL (`NERVA_SYCL`) | unchanged | disabled | disabled |
| thread control (`set_num_threads`) | `mkl_set_num_threads` | `openblas_set_num_threads`/`bli_thread_set_num_threads` if available, plus `omp_set_num_threads` | `Eigen::setNbThreads` / `omp_set_num_threads` |

All of these are templates/inline functions in headers selected by `#if`, so
there is no run-time dispatch.

### 3.5 The `--computation` modes

`computation::{eigen, mkl, blas, sycl}` stays as is, so command lines and
scripts keep working. In builds without the corresponding backend:

* `mkl` maps to the backend's `ddd_product` (i.e. "explicit GEMM call instead
  of an Eigen expression"); this preserves the reason the mode exists – the
  `softmax_layer` backprop workaround described in `performance.adoc` – while
  not requiring MKL.
* `blas` behaves the same as `mkl` when no BLAS is linked.
* `sycl` keeps throwing `"computation mode sycl is disabled"`.

`set_nerva_computation` gains a function `nerva_backend()` (and a Python
binding) that returns `"mkl"`, `"blas"` or `"eigen"`, so that logs and
benchmark output record which backend produced a number.

### 3.6 Tests, tools and examples

* `mkl_test.cpp`, `mkl_matrix_test.cpp`, `sycl_test.cpp`, `tools/mkl.cpp`,
  `examples/subnormal_numbers.cpp`: built only when `NERVA_BACKEND=mkl`
  (VML and MKL-specific comparisons). The backend-neutral parts of
  `mkl_matrix_test.cpp` (products, CSR conversions, pruning) move into a new
  `linalg_test.cpp` that runs for every backend, comparing each kernel against
  a naive Eigen reference.
* CI: add a matrix job `{backend: eigen, blas(OpenBLAS)}` on
  `ubuntu-latest` and `macos-latest`; the MKL job stays as it is. This is the
  only way to keep the non-MKL path from bit-rotting.

## 4. Why this has no runtime cost for MKL builds

* The MKL build compiles the same function bodies; they only live in a
  different header. `ddd_product`, `dds_product`, `sdd_product_batch`,
  `construct_csr` and the optimizer updates expand to the same MKL calls.
* `sparse_matrix_csr` keeps the same members in the same order; the handle is
  just wrapped in a struct, which is layout-identical.
* No new virtual functions, function pointers or run-time checks are
  introduced. The only run-time branches are the existing
  `NervaComputation` tests.
* `EIGEN_USE_MKL_ALL` and `MKL_ILP64` stay defined exactly as now.

Verification step (part of the implementation, see section 7): build `mlp`
before and after the refactoring with `NERVA_BACKEND=mkl`, compare
`objdump -d` of the hot functions (`linear_layer::feedforward`,
`backpropagate`, `momentum_optimizer::update`), and run the standard MNIST /
CIFAR-10 benchmarks from `nerva-cpp-performance.adoc`; the timings must be
within noise.

## 5. Alternative backends for the expensive operations

### 5.1 Dense GEMM (`ddd_product`)

| Library | License | Platforms | Notes |
|---|---|---|---|
| **Eigen** (built-in) | MPL2 | all | Good GEMM, multithreaded with OpenMP; typically 1.2–2× slower than MKL for large SGEMM on x86, competitive for the small/medium sizes of MLP batches. Zero extra dependencies. |
| **OpenBLAS** | BSD | all, incl. ARM64 / RISC-V | Closest to MKL on x86 and the best general choice for a `blas` backend; packaged everywhere (`libopenblas-dev`, Homebrew, conda, vcpkg). Has `cblas_?omatcopy`/`?imatcopy` and `cblas_i?amin` extensions. ILP64 builds exist (`openblas64`). |
| **BLIS / AMD AOCL-BLIS** | BSD | all | Very good on AMD Zen; typed and object API plus CBLAS compatibility layer. |
| **Apple Accelerate** | system | macOS | Uses the AMX units on Apple Silicon; by far the fastest option on M-series Macs. CBLAS interface (`-framework Accelerate`, new LAPACK headers since macOS 13.3). |
| **Arm Performance Libraries** | free, closed | Linux/macOS ARM64 | BLAS + sparse; relevant for Graviton/Grace servers. |
| **cuBLAS / hipBLAS** | vendor | NVIDIA / AMD GPUs | Would require device-resident matrices; out of scope for this plan (see 5.4). |

Because all of these export the standard CBLAS interface, a single `blas`
backend covers them; the choice is made at link time
(`-DBLA_VENDOR=OpenBLAS|FLAME|Apple|Arm_mp` with CMake's `FindBLAS`).

### 5.2 Sparse × dense (`dds_product`, SpMM)

This is the kernel where MKL is hardest to replace, and the one that matters
most for sparse training.

| Library | Notes |
|---|---|
| **Eigen `SparseMatrix<RowMajor>` × dense** | Zero-copy over the existing CSR arrays, OpenMP-parallel over rows. Expected to be noticeably slower than `mkl_sparse_?_mm` (MKL uses an inspector-executor with optimised kernels); must be measured. This is the default for the `blas` and `eigen` backends. |
| **Hand-written CSR SpMM kernel** | For `Y = X · Wᵀ` with `X` row-major dense and `W` CSR, a row-parallel kernel (OpenMP over the rows of `W`, inner loop an `axpy` over a contiguous row of `X`ᵀ / column of `Y`) is ~50 lines and vectorises well. Worth benchmarking against Eigen; it can live in `backend_eigen.h` with no extra dependency. |
| **AMD AOCL-Sparse** | Open source (MIT), CSR SpMM (`aoclsparse_?csrmm`) with an MKL-like inspector-executor API; maps almost 1:1 on `sparse_matrix_csr` + handle. Best candidate for a second "native sparse" backend on x86. |
| **Arm Performance Libraries sparse** | `armpl_spmm_exec_?`, also inspector-executor; ARM64 only. |
| **Apple Accelerate Sparse (`SparseMultiply`)** | Supports CSC/CSR-like `SparseMatrix_Float/Double` × dense; adequate on macOS. |
| **SuiteSparse:GraphBLAS** | Very fast `mxm` with semirings, JIT-compiled kernels, Apache-2 license; heavier dependency, own matrix objects (import/export of CSR is O(1) with `GxB_Matrix_pack_CSR`). |
| **librsb** | Recursive sparse blocks, good multicore SpMM; LGPL. |
| **oneMath (formerly oneMKL Interfaces)** | Open-source SYCL front-end with MKL, cuSPARSE and rocSPARSE back-ends; relevant only together with the existing SYCL experiments. |

Recommendation: implement the Eigen path first (it is needed anyway as the
universal fallback), add the hand-written kernel if Eigen is clearly behind,
and treat AOCL-Sparse / ArmPL as optional extra backends behind the same
`backend::csr_handle` + `dds_product` interface only if benchmarks justify it.
The interface in section 3.3 (CSR arrays + opaque, rebuildable handle) was
chosen precisely so that inspector-executor libraries slot in the same way MKL
does.

### 5.3 SDDMM (`sdd_product_batch`)

The current implementation is already library-agnostic: it computes dense
blocks with GEMM and gathers the values on the sparsity pattern. It therefore
inherits whatever GEMM the backend uses. A dedicated SDDMM kernel (dot product
per nonzero, `O(nnz · batch)` instead of `O(rows · cols · batch)`) becomes
faster than the GEMM approach at high sparsity (roughly density < 5–10 %);
that is an independent optimisation that also benefits the MKL build and is
not part of making MKL optional.

### 5.4 GPUs

cuBLAS/cuSPARSE (or hipBLAS/rocSPARSE, or oneMath via SYCL) would provide
all three kernels, but only pay off if activations and weights stay on the
device across layers. That is an architectural change (device-resident
`eigen::matrix` replacements) and is out of scope; the backend split proposed
here does not preclude it.

## 6. Expected performance without MKL (to be measured)

* Dense MLPs: Eigen-only within a small factor of MKL; OpenBLAS / BLIS /
  Accelerate close to MKL. Note that Eigen's own `exp`/`log`/`tanh` replace
  MKL VML in activations; they are SIMD-vectorised and should be close.
* Sparse MLPs: dominated by SpMM; Eigen likely 1.5–3× slower than MKL. This
  is acceptable for a fallback but should be documented in
  `nerva-cpp-performance.adoc` with actual numbers.
* `--computation mkl` workaround for `softmax_layer`: still effective with the
  Eigen/BLAS `ddd_product`, because the gain comes from avoiding a poor Eigen
  expression evaluation, not from MKL itself. To be confirmed by benchmark.

## 7. Implementation steps

Each step keeps the MKL build bit-for-bit equivalent in behaviour and is a
separate commit in both `nerva-rowwise` and `nerva-colwise` (do rowwise first,
then port; the diff between the two repositories in these headers is small
and layout-related).

1. **Baseline.** Record MKL timings (MNIST/CIFAR-10, dense and sparse at
   densities 1.0, 0.1, 0.01) and keep the `mlp` binary for disassembly
   comparison.
2. **Index type.** Introduce `linalg::index_type`; replace `MKL_INT` outside
   the MKL backend (`mkl_sparse_matrix.h`, `csr_matrix_builder`,
   `mlp_algorithms.h`). No functional change.
3. **Split headers.** Move code into `linalg/` as in 3.2, MKL code verbatim
   into `backend_mkl.h`; keep `mkl_*.h` as compatibility includes and
   `namespace mkl = linalg`. MKL-only build still. Verify 1. is unchanged.
4. **CSR handle.** Extract `backend::csr_handle<T>` from
   `sparse_matrix_csr` (3.3). Verify unchanged.
5. **Eigen backend.** Implement `backend_eigen.h` (3.4) and the
   `NERVA_BACKEND` option in CMake with `eigen` and `mkl`. Add
   `linalg_test.cpp`. Make all tests pass in both configurations.
6. **BLAS backend.** Add `backend_blas.h` (CBLAS GEMM + Eigen sparse) and the
   `blas` option; test with OpenBLAS on Linux and Accelerate on macOS.
7. **Python packaging.** `setup.py` backend switch; remove the hard MKL
   requirement; expose `nerva_backend()`; replace `set_num_threads`.
8. **Build files & CI.** Guard MKL-only tests/tools/examples; CI matrix for
   the non-MKL backends; optional `Jamroot` support.
9. **Documentation.** Update `installation.adoc` (C++ and Python),
   `matrix-operations.adoc`, `performance.adoc` and the READMEs: MKL becomes
   "recommended for best performance on x86" instead of "required"; publish
   the benchmark table from step 1 vs. steps 5/6.
10. **Optional.** Hand-written CSR SpMM kernel and/or AOCL-Sparse backend if
    the step-9 numbers show the Eigen SpMM to be the bottleneck.

## 8. Risks and open questions

* **Eigen sparse map semantics.** `Eigen::Map<SparseMatrix>` requires
  `index_type` to be a signed integer and the outer index array of size
  `rows + 1` — both already true for the current CSR layout
  (`m_row_index.data()`, `m_row_index.data() + 1` is exactly what MKL gets).
* **ILP64 vs. LP64 BLAS.** The `blas` backend passes `int` dimensions to
  standard CBLAS; matrices above 2³¹ elements would need an ILP64 OpenBLAS.
  Not an issue for current model sizes, but add an assertion.
* **Threading.** MKL uses its own thread pool (`mkl_intel_thread`); Eigen and
  OpenBLAS use OpenMP/pthreads. Mixing an OpenMP Eigen with a pthreads
  OpenBLAS can oversubscribe; document using the OpenMP build of OpenBLAS or
  `OPENBLAS_NUM_THREADS=1` when Eigen's own parallelism is used.
* **Numerical differences.** Different summation orders give results that
  differ in the last bits; tests that compare against stored reference
  outputs need tolerances rather than exact equality.
* **colwise vs. rowwise.** In `nerva-colwise` the sparse product is
  `Y = W · X` (`dsd_product` rather than `dds_product` with transposed `W`);
  with a row-major CSR map this is the natural, fast direction for Eigen,
  so the colwise library is expected to lose less performance without MKL
  than the rowwise one.
