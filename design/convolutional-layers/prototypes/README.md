# Prototype scripts for the convolutional-layer design

These scripts were written during the design investigation to check the equations and to measure performance. They are **not** part of the Nerva libraries. They only need NumPy: they were run with NumPy 2.5.3 and Python 3.13 on 4 CPU cores.

Run them from this directory, for example `python3 rev3_pixel_matrix.py`.

| Script | Design revision | What it checks or measures |
|---|---|---|
| `rev1_im2col_reference.py` | 1 (one row per image, (p,q,c) columns) | im2col convolution against direct loops; finite differences for DX, DW, Db; adjointness of im2col/col2im; max pooling; exactness with integer inputs; float32 rounding error; timings |
| `rev1_benchmark.py` | 1 | im2col against shift-and-accumulate (one product per kernel position) |
| `rev2_primitive_layers.py` | 2 (layers in primitives incl. `reshape`, (c,p,q) columns) | Conv, max-pool and average-pool layers written in primitives; finite differences; tie handling |
| `rev3_pixel_matrix.py` | 3 (pixel matrix, selection matrices `S_pq`, no reshape) | Builds `S_pq` explicitly and checks that the strided-slice im2col/col2im equal the definitions; layers; finite differences; flatten via a full-image window; prints the worked example of the specification |
| `rev3_benchmark_selection_matrices.py` | 3 | Strided slices against gather/scatter (sparse `S_pq`) against the memory dense `S_pq` would need; matrix-product times |
| `rev3_benchmark_optimized.py` | 3 | Optimised strided-slice im2col/col2im and an argmax-based `blocks_max_mask` |

Timings vary from run to run by up to about 2×. Only the orders of magnitude matter.

Inside these prototypes, `reshape` is used freely as an *implementation technique*: it views a pixel matrix's memory as an N×H×W×C array. The design keeps it out of the layer equations.
