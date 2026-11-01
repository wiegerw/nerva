# Prototype (not part of the Nerva libraries): revision 1 of the design. im2col with (p,q,c) columns on one-row-per-image input.
import numpy as np, time
from numpy.lib.stride_tricks import sliding_window_view as swv

def out_size(H, k, s, d, p0, p1):
    return (H + p0 + p1 - d*(k-1) - 1)//s + 1

class G:  # geometry
    def __init__(s, H, W, C, K, kh, kw, sh=1, sw=1, dh=1, dw=1, pt=0, pb=0, pl=0, pr=0):
        s.__dict__.update(locals()); del s.__dict__['s']
        s.Ho = out_size(H, kh, sh, dh, pt, pb); s.Wo = out_size(W, kw, sw, dw, pl, pr)

def im2col(X, g):           # X: (N, H*W*C) -> (N*Ho*Wo, kh*kw*C), column order (p, q, c)
    N = X.shape[0]
    X4 = X.reshape(N, g.H, g.W, g.C)
    Xp = np.pad(X4, ((0,0),(g.pt,g.pb),(g.pl,g.pr),(0,0)))
    cols = []
    for p in range(g.kh):
        for q in range(g.kw):
            h0, w0 = p*g.dh, q*g.dw
            cols.append(Xp[:, h0:h0+g.sh*(g.Ho-1)+1:g.sh, w0:w0+g.sw*(g.Wo-1)+1:g.sw, :])
    P = np.stack(cols, axis=3)  # (N, Ho, Wo, kh*kw, C)
    return P.reshape(N*g.Ho*g.Wo, g.kh*g.kw*g.C)

def col2im(P, g, N):        # adjoint of im2col
    Xp = np.zeros((N, g.H+g.pt+g.pb, g.W+g.pl+g.pr, g.C), P.dtype)
    P5 = P.reshape(N, g.Ho, g.Wo, g.kh*g.kw, g.C)
    for p in range(g.kh):
        for q in range(g.kw):
            h0, w0 = p*g.dh, q*g.dw
            Xp[:, h0:h0+g.sh*(g.Ho-1)+1:g.sh, w0:w0+g.sw*(g.Wo-1)+1:g.sw, :] += P5[:, :, :, p*g.kw+q, :]
    return Xp[:, g.pt:g.pt+g.H, g.pl:g.pl+g.W, :].reshape(N, -1)

def conv_fwd(X, W, b, g):   # W: (K, kh*kw*C); returns (N, Ho*Wo*K) in NHWC-flat
    N = X.shape[0]
    Z = im2col(X, g) @ W.T + b
    return Z.reshape(N, -1)

def conv_bwd(DZ, X, W, g):
    N = X.shape[0]
    DZm = DZ.reshape(N*g.Ho*g.Wo, g.K)
    DW = DZm.T @ im2col(X, g)
    Db = DZm.sum(axis=0)
    DX = col2im(DZm @ W, g, N)
    return DX, DW, Db

def conv_naive(X, W, b, g):
    N = X.shape[0]; X4 = X.reshape(N, g.H, g.W, g.C); W4 = W.reshape(g.K, g.kh, g.kw, g.C)
    Z = np.zeros((N, g.Ho, g.Wo, g.K))
    for n in range(N):
        for i in range(g.Ho):
            for j in range(g.Wo):
                for k in range(g.K):
                    s = b[k]
                    for p in range(g.kh):
                        for q in range(g.kw):
                            h = g.sh*i + g.dh*p - g.pt; w = g.sw*j + g.dw*q - g.pl
                            if 0 <= h < g.H and 0 <= w < g.W:
                                s += W4[k, p, q] @ X4[n, h, w]
                    Z[n, i, j, k] = s
    return Z.reshape(N, -1)

def maxpool(X, H, W, C, k, s):
    N = X.shape[0]; X4 = X.reshape(N, H, W, C)
    Ho, Wo = (H-k)//s+1, (W-k)//s+1
    win = swv(X4, (k, k), axis=(1, 2))[:, ::s, ::s][:, :Ho, :Wo]  # (N,Ho,Wo,C,k,k)
    flat = win.reshape(N, Ho, Wo, C, k*k)
    idx = flat.argmax(axis=-1)    # first maximum in row-major window order
    Y = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
    return Y.reshape(N, -1), idx

def maxpool_bwd(DY, idx, N, H, W, C, k, s):
    Ho, Wo = idx.shape[1:3]; DX = np.zeros((N, H, W, C)); DY4 = DY.reshape(N, Ho, Wo, C)
    for p in range(k):
        for q in range(k):
            m = (idx == p*k+q)
            DX[:, p:p+s*(Ho-1)+1:s, q:q+s*(Wo-1)+1:s, :] += DY4 * m
    return DX.reshape(N, -1)

rng = np.random.default_rng(1)
def fd_check(f, x, Dx, h=1e-6, nprobe=25):
    err = 0
    for _ in range(nprobe):
        i = tuple(rng.integers(0, s) for s in x.shape)
        old = x[i]; x[i] = old + h; fp = f(); x[i] = old - h; fm = f(); x[i] = old
        err = max(err, abs((fp - fm)/(2*h) - Dx[i]))
    return err

if __name__ == '__main__':
    configs = [G(5,6,2,3,3,3), G(7,7,3,4,3,2,sh=2,sw=1,pt=1,pb=1,pl=1,pr=0), G(9,8,2,2,3,3,dh=2,dw=2,pt=2,pb=2,pl=2,pr=2),
               G(6,6,1,2,2,2,sh=3,sw=3), G(8,8,3,5,3,3,sh=2,sw=2,pt=1,pb=0,pl=1,pr=0)]
    for g in configs:
        N = 2
        X = rng.standard_normal((N, g.H*g.W*g.C)); W = rng.standard_normal((g.K, g.kh*g.kw*g.C)); b = rng.standard_normal(g.K)
        Z = conv_fwd(X, W, b, g)
        fwd = np.abs(Z - conv_naive(X, W, b, g)).max()
        R = rng.standard_normal(Z.shape)      # loss = <R, Z>
        DX, DW, Db = conv_bwd(R, X, W, g)
        L = lambda: np.sum(R * conv_fwd(X, W, b, g))
        e = (fd_check(L, X, DX), fd_check(L, W, DW), fd_check(L, b, Db))
        P = rng.standard_normal((N*g.Ho*g.Wo, g.kh*g.kw*g.C))
        adj = abs(np.sum(im2col(X, g)*P) - np.sum(X*col2im(P, g, N)))
        print(f'H={g.H} W={g.W} C={g.C} K={g.K} k=({g.kh},{g.kw}) s=({g.sh},{g.sw}) d=({g.dh},{g.dw}) pad=({g.pt},{g.pb},{g.pl},{g.pr}) -> Ho={g.Ho} Wo={g.Wo}: fwd {fwd:.1e} fd(DX,DW,Db) {e[0]:.1e} {e[1]:.1e} {e[2]:.1e} adjoint {adj:.1e}')

    # pooling (distinct values, no ties)
    N,H,W,C,k,s = 2,6,6,3,2,2
    X = rng.permutation(N*H*W*C).astype(float).reshape(N, -1) / 7
    Y, idx = maxpool(X, H, W, C, k, s); R = rng.standard_normal(Y.shape)
    DX = maxpool_bwd(R, idx, N, H, W, C, k, s)
    print('maxpool fd err', fd_check(lambda: np.sum(R*maxpool(X,H,W,C,k,s)[0]), X, DX, h=1e-4))
    # overlapping windows k=3 s=2
    Y, idx = maxpool(X, H, W, C, 3, 2); R = rng.standard_normal(Y.shape)
    DX = maxpool_bwd(R, idx, N, H, W, C, 3, 2)
    print('maxpool overlap fd err', fd_check(lambda: np.sum(R*maxpool(X,H,W,C,3,2)[0]), X, DX, h=1e-4), 'Y shape', Y.shape)

    # integer-valued float32 exactness: conv on small ints gives bit-exact results in float32 vs float64
    g = G(8,8,3,4,3,3,pt=1,pb=1,pl=1,pr=1)
    Xi = rng.integers(-3, 4, (2, 192)); Wi = rng.integers(-3, 4, (4, 27)); bi = rng.integers(-3, 4, 4)
    z32 = conv_fwd(Xi.astype(np.float32), Wi.astype(np.float32), bi.astype(np.float32), g)
    z64 = conv_fwd(Xi.astype(np.float64), Wi.astype(np.float64), bi.astype(np.float64), g)
    print('integer exactness float32 vs float64 max diff:', np.abs(z32 - z64).max())

    # float32 vs float64 error for realistic sizes
    g = G(16,16,32,64,3,3,pt=1,pb=1,pl=1,pr=1)
    X = rng.standard_normal((4, 16*16*32)); W = rng.standard_normal((64, 288))*np.sqrt(2/288); b = np.zeros(64)
    z64 = conv_fwd(X, W, b, g); z32 = conv_fwd(X.astype(np.float32), W.astype(np.float32), b.astype(np.float32), g)
    bound = conv_fwd(np.abs(X), np.abs(W), b, g)
    print('float32 rel err (max |dz|/max|z|):', np.abs(z32-z64).max()/np.abs(z64).max(), ' max |dz|/(F*eps*|X|*|W|):', (np.abs(z32-z64)/(288*np.finfo(np.float32).eps*bound)).max())

    # memory/timing for CIFAR batch of 100, float32
    for (H,C,K) in [(32,3,32),(16,32,64)]:
        g = G(H,H,C,K,3,3,pt=1,pb=1,pl=1,pr=1); N = 100
        X = rng.standard_normal((N, H*H*C)).astype(np.float32); W = rng.standard_normal((K, 9*C)).astype(np.float32); b = np.zeros(K, np.float32)
        t = time.perf_counter(); P = im2col(X, g); t1 = time.perf_counter()-t
        t = time.perf_counter(); Z = conv_fwd(X, W, b, g); t2 = time.perf_counter()-t
        t = time.perf_counter(); conv_bwd(Z, X, W, g); t3 = time.perf_counter()-t
        print(f'conv H={H} C={C} K={K} N={N}: im2col {P.nbytes/2**20:.1f} MiB (input {X.nbytes/2**20:.1f} MiB), im2col {t1*1e3:.1f} ms, fwd {t2*1e3:.1f} ms, bwd {t3*1e3:.1f} ms')
