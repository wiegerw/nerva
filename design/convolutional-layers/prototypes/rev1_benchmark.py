# Benchmark (revision 1): im2col versus shift-and-accumulate, NumPy.
import numpy as np, time, os
from rev1_im2col_reference import G, im2col, conv_fwd, conv_bwd
def conv_shift(X, W, b, g):
    N = X.shape[0]; Xp = np.pad(X.reshape(N,g.H,g.W,g.C), ((0,0),(g.pt,g.pb),(g.pl,g.pr),(0,0)))
    W4 = W.reshape(g.K, g.kh, g.kw, g.C); Z = np.empty((N, g.Ho, g.Wo, g.K), X.dtype); Z[:] = b
    for p in range(g.kh):
        for q in range(g.kw):
            S = Xp[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :]
            Z += S @ W4[:, p, q, :].T
    return Z.reshape(N, -1)
def t(f, r=5):
    f(); s = time.perf_counter()
    for _ in range(r): f()
    return (time.perf_counter()-s)/r*1e3
rng = np.random.default_rng(0)
print('cpus', os.cpu_count())
for (H,C,K) in [(32,3,32),(16,32,64)]:
    g = G(H,H,C,K,3,3,pt=1,pb=1,pl=1,pr=1); N = 100
    X = rng.standard_normal((N, H*H*C)).astype(np.float32); W = rng.standard_normal((K, 9*C)).astype(np.float32); b = np.zeros(K, np.float32)
    assert np.allclose(conv_shift(X,W,b,g), conv_fwd(X,W,b,g), atol=1e-4)
    Z = conv_fwd(X,W,b,g)
    print(f'H={H} C={C} K={K}: im2col-fwd {t(lambda: conv_fwd(X,W,b,g)):.1f} ms, shift-fwd {t(lambda: conv_shift(X,W,b,g)):.1f} ms, im2col-bwd {t(lambda: conv_bwd(Z,X,W,g)):.1f} ms, '
          f'GFLOP fwd {2*N*g.Ho*g.Wo*K*9*C/1e9:.2f}')
