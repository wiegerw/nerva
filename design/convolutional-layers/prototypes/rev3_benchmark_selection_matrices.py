# Benchmark (revision 3): strided slices versus gather/scatter (sparse S_pq) versus dense S_pq memory, NumPy.
import numpy as np, time
from rev3_pixel_matrix import G, im2col, col2im, blocks_max, blocks_max_mask, blocks_repeat
rng = np.random.default_rng(0)
def t(f, r=5):
    f(); s = time.perf_counter()
    for _ in range(r): f()
    return (time.perf_counter() - s) / r * 1e3

def index_tables(g):
    """For each (p,q): source row index into X for every output row; -1 means padding (zero row of S_pq)."""
    n, i, j = np.meshgrid(np.arange(g.N), np.arange(g.Ho), np.arange(g.Wo), indexing='ij')
    tabs = []
    for p in range(g.kh):
        for q in range(g.kw):
            h = g.sh*i + g.dh*p - g.pt; w = g.sw*j + g.dw*q - g.pl
            ok = (h >= 0) & (h < g.H) & (w >= 0) & (w < g.W)
            tabs.append(np.where(ok, (n*g.H + h)*g.W + w, -1).ravel())
    return tabs
def im2col_gather(X, tabs):
    Xz = np.vstack([X, np.zeros((1, X.shape[1]), X.dtype)])          # row -1 -> zero row
    return np.concatenate([Xz[tb] for tb in tabs], axis=1)
def col2im_scatter(P, tabs, nrows):
    C = P.shape[1] // len(tabs); out = np.zeros((nrows + 1, C), P.dtype)
    for b, tb in enumerate(tabs):
        np.add.at(out, tb, P[:, b*C:(b+1)*C])
    return out[:-1]

N = 100
print(f'{"layer":28s} {"im2col":>8s} {"GEMM fwd":>9s} {"GEMM bwd(2)":>12s} {"col2im":>8s} | {"gather":>7s} {"scatter":>8s} | dense S_pq would need')
for name, (H, C, K) in [('conv1 32x32x3 -> 32', (32, 3, 32)), ('conv2 16x16x32 -> 64', (16, 32, 64))]:
    g = G(N, H, H, 3, 3, pt=1, pb=1, pl=1, pr=1)
    X = rng.standard_normal((N*H*H, C)).astype(np.float32); W = rng.standard_normal((K, 9*C)).astype(np.float32)
    Xc = im2col(X, g); DZ = rng.standard_normal((Xc.shape[0], K)).astype(np.float32); Q = DZ @ W
    tabs = index_tables(g)
    assert np.array_equal(im2col_gather(X, tabs), Xc) and np.allclose(col2im_scatter(Q, tabs, N*H*H), col2im(Q, g), atol=1e-4)
    dense = 9 * (N*H*H)**2 * 4 / 2**30
    print(f'{name:28s} {t(lambda: im2col(X, g)):7.1f}ms {t(lambda: Xc @ W.T):8.1f}ms {t(lambda: (DZ.T @ Xc, DZ @ W)):11.1f}ms {t(lambda: col2im(Q, g)):7.1f}ms |'
          f' {t(lambda: im2col_gather(X, tabs)):6.1f}ms {t(lambda: col2im_scatter(Q, tabs, N*H*H), 2):7.1f}ms | {dense:,.0f} GiB')
g = G(N, 32, 32, 2, 2, sh=2, sw=2); X = rng.standard_normal((N*32*32, 32)).astype(np.float32)
P = im2col(X, g); Y = blocks_max(P, 4)
print(f'maxpool 32x32x32: im2col {t(lambda: im2col(X, g)):.1f} ms, blocks_max {t(lambda: blocks_max(P, 4)):.1f} ms, '
      f'backward {t(lambda: col2im(blocks_max_mask(P, 4) * blocks_repeat(Y, 4), g)):.1f} ms')
