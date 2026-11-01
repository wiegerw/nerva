# Benchmark (revision 3): optimized strided-slice im2col/col2im and an argmax-based blocks_max_mask, NumPy.
import numpy as np, time
from rev3_pixel_matrix import G, im2col, col2im, blocks_max, blocks_max_mask, blocks_repeat
def t(f, r=5):
    f(); s = time.perf_counter()
    for _ in range(r): f()
    return (time.perf_counter() - s) / r * 1e3
def im2col_fast(X, g):
    C = X.shape[1]; Xp = np.pad(X.reshape(g.N, g.H, g.W, C), ((0,0),(g.pt,g.pb),(g.pl,g.pr),(0,0)))
    out = np.empty((g.N, g.Ho, g.Wo, g.kh, g.kw, C), X.dtype)
    for p in range(g.kh):
        for q in range(g.kw):
            out[:, :, :, p, q, :] = Xp[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :]
    return out.reshape(g.N*g.Ho*g.Wo, g.kh*g.kw*C)
def col2im_fast(P, g):
    C = P.shape[1]//(g.kh*g.kw); P6 = P.reshape(g.N, g.Ho, g.Wo, g.kh, g.kw, C)   # view, no copy
    Xp = np.zeros((g.N, g.H+g.pt+g.pb, g.W+g.pl+g.pr, C), P.dtype)
    for p in range(g.kh):
        for q in range(g.kw):
            Xp[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :] += P6[:, :, :, p, q, :]
    return Xp[:, g.pt:g.pt+g.H, g.pl:g.pl+g.W, :].reshape(-1, C)
def blocks_max_mask_fast(P, B):   # argmax-based: NumPy's argmax returns the first maximum
    M, BC = P.shape; C = BC // B; P3 = P.reshape(M, B, C)
    idx = P3.argmax(axis=1); E = np.zeros_like(P3); np.put_along_axis(E, idx[:, None, :], 1, axis=1)
    return E.reshape(M, BC)
rng = np.random.default_rng(0); N = 100
for name, (H, C, K) in [('conv1', (32, 3, 32)), ('conv2', (16, 32, 64))]:
    g = G(N, H, H, 3, 3, pt=1, pb=1, pl=1, pr=1)
    X = rng.standard_normal((N*H*H, C)).astype(np.float32); W = rng.standard_normal((K, 9*C)).astype(np.float32)
    Xc = im2col(X, g); Q = rng.standard_normal(Xc.shape).astype(np.float32)
    assert np.array_equal(im2col_fast(X, g), Xc) and np.allclose(col2im_fast(Q, g), col2im(Q, g), atol=1e-4)
    print(f'{name}: im2col {t(lambda: im2col(X, g)):.1f} -> {t(lambda: im2col_fast(X, g)):.1f} ms,  col2im {t(lambda: col2im(Q, g)):.1f} -> {t(lambda: col2im_fast(Q, g)):.1f} ms')
g = G(N, 32, 32, 2, 2, sh=2, sw=2); X = rng.standard_normal((N*32*32, 32)).astype(np.float32); P = im2col_fast(X, g)
assert np.array_equal(blocks_max_mask(P, 4), blocks_max_mask_fast(P, 4))
Y = blocks_max(P, 4)
print(f'maxpool: mask {t(lambda: blocks_max_mask(P, 4)):.1f} -> {t(lambda: blocks_max_mask_fast(P, 4)):.1f} ms, '
      f'backward total {t(lambda: col2im_fast(blocks_max_mask_fast(P, 4) * blocks_repeat(Y, 4), g)):.1f} ms')
