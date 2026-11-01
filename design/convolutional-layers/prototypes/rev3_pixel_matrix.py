# Prototype (not part of the Nerva libraries): revision 3 of the design. Pixel-matrix representation, selection matrices S_pq, no reshape in layer code.
import numpy as np
rng = np.random.default_rng(5)

class G:
    def __init__(s, N, H, W, kh, kw, sh=1, sw=1, dh=1, dw=1, pt=0, pb=0, pl=0, pr=0):
        s.__dict__.update({k: v for k, v in locals().items() if k != 's'})
        s.Ho = (H + pt + pb - dh*(kh-1) - 1)//sh + 1
        s.Wo = (W + pl + pr - dw*(kw-1) - 1)//sw + 1

def S(g, p, q):
    """Selection matrix S_pq in {0,1}^{(N*Ho*Wo) x (N*H*W)} (dense, for the specification only)."""
    M = np.zeros((g.N*g.Ho*g.Wo, g.N*g.H*g.W))
    for n in range(g.N):
        for i in range(g.Ho):
            for j in range(g.Wo):
                h, w = g.sh*i + g.dh*p - g.pt, g.sw*j + g.dw*q - g.pl
                if 0 <= h < g.H and 0 <= w < g.W:
                    M[(n*g.Ho + i)*g.Wo + j, (n*g.H + h)*g.W + w] = 1
    return M

# --- specification versions (only matrix algebra) ---
def im2col_spec(X, g):  return np.hstack([S(g, p, q) @ X for p in range(g.kh) for q in range(g.kw)])
def col2im_spec(P, g):
    C = P.shape[1] // (g.kh*g.kw)
    return sum(S(g, p, q).T @ P[:, (p*g.kw+q)*C:(p*g.kw+q+1)*C] for p in range(g.kh) for q in range(g.kw))

# --- block primitives ---
def blocks(P, B): C = P.shape[1]//B; return [P[:, b*C:(b+1)*C] for b in range(B)]
def blocks_sum(P, B): return sum(blocks(P, B))
def blocks_repeat(Y, B): return np.hstack([Y]*B)
def blocks_max(P, B): return np.max(np.stack(blocks(P, B)), axis=0)
def blocks_max_mask(P, B):
    st = np.stack(blocks(P, B)); E = st == st.max(axis=0); F = E & (np.cumsum(E, axis=0) == 1)
    return np.hstack(list(F.astype(P.dtype)))

# --- efficient im2col/col2im (strided slices), must equal spec ---
def im2col(X, g):
    C = X.shape[1]; X4 = np.pad(X.reshape(g.N, g.H, g.W, C), ((0,0),(g.pt,g.pb),(g.pl,g.pr),(0,0)))
    return np.concatenate([X4[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :]
                           .reshape(-1, C) for p in range(g.kh) for q in range(g.kw)], axis=1)
def col2im(P, g):
    C = P.shape[1]//(g.kh*g.kw); Xp = np.zeros((g.N, g.H+g.pt+g.pb, g.W+g.pl+g.pr, C))
    for p in range(g.kh):
        for q in range(g.kw):
            b = p*g.kw + q
            Xp[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :] += P[:, b*C:(b+1)*C].reshape(g.N, g.Ho, g.Wo, C)
    return Xp[:, g.pt:g.pt+g.H, g.pl:g.pl+g.W, :].reshape(-1, C)

row_repeat = lambda b, m: np.tile(b[None, :], (m, 1)); columns_sum = lambda X: X.sum(0); hadamard = lambda A, B: A*B

# --- layers: only primitives, no reshape ---
def conv_ff(X, W, b, g): Xc = im2col(X, g); return Xc @ W.T + row_repeat(b, g.N*g.Ho*g.Wo), Xc
def conv_bp(DZ, Xc, W, g): return DZ.T @ Xc, columns_sum(DZ), col2im(DZ @ W, g)
def maxpool_ff(X, g): P = im2col(X, g); return blocks_max(P, g.kh*g.kw), P
def maxpool_bp(DY, P, g): B = g.kh*g.kw; return col2im(hadamard(blocks_max_mask(P, B), blocks_repeat(DY, B)), g)
def avgpool_ff(X, g): B = g.kh*g.kw; return blocks_sum(im2col(X, g), B) / B
def avgpool_bp(DY, g): B = g.kh*g.kw; return col2im(blocks_repeat(DY, B), g) / B

def fd(L, x, D, h=1e-6, n=30):
    e = 0
    for _ in range(n):
        i = tuple(rng.integers(0, s) for s in x.shape); o = x[i]
        x[i] = o + h; a = L(); x[i] = o - h; c = L(); x[i] = o; e = max(e, abs((a-c)/(2*h) - D[i]))
    return e

if __name__ == '__main__':
    for g, C, K in [(G(2,5,6,3,3), 2, 3), (G(2,7,7,3,2,sh=2,pt=1,pb=1,pl=1), 3, 4), (G(2,9,8,3,3,dh=2,dw=2,pt=2,pb=2,pl=2,pr=2), 2, 2), (G(2,6,6,2,2,sh=3,sw=3), 1, 2)]:
        X = rng.standard_normal((g.N*g.H*g.W, C)); W = rng.standard_normal((K, g.kh*g.kw*C)); b = rng.standard_normal(K)
        P = rng.standard_normal((g.N*g.Ho*g.Wo, g.kh*g.kw*C))
        spec = max(np.abs(im2col(X, g) - im2col_spec(X, g)).max(), np.abs(col2im(P, g) - col2im_spec(P, g)).max())
        # naive conv
        X4 = X.reshape(g.N, g.H, g.W, C); W4 = W.reshape(K, g.kh, g.kw, C); Zn = np.zeros((g.N, g.Ho, g.Wo, K))
        for n in range(g.N):
            for i in range(g.Ho):
                for j in range(g.Wo):
                    for p in range(g.kh):
                        for q in range(g.kw):
                            h, w = g.sh*i+g.dh*p-g.pt, g.sw*j+g.dw*q-g.pl
                            if 0 <= h < g.H and 0 <= w < g.W: Zn[n,i,j] += W4[:,p,q] @ X4[n,h,w]
        Z, Xc = conv_ff(X, W, b, g); fwd = np.abs(Z - (Zn + b).reshape(-1, K)).max()
        R = rng.standard_normal(Z.shape); DW, Db, DX = conv_bp(R, Xc, W, g); L = lambda: np.sum(R*conv_ff(X, W, b, g)[0])
        print(f'conv: spec==impl {spec:.0e}, fwd {fwd:.0e}, fd DX {fd(L,X,DX):.0e} DW {fd(L,W,DW):.0e} Db {fd(L,b,Db):.0e}')

    for k, s in [(2,2),(3,2),(3,1)]:
        g = G(2, 6, 7, k, k, sh=s, sw=s); C = 3
        X = (rng.permutation(2*6*7*C)/11.0).reshape(-1, C)
        Y, P = maxpool_ff(X, g); R = rng.standard_normal(Y.shape); DX = maxpool_bp(R, P, g)
        Ya = avgpool_ff(X, g); DXa = avgpool_bp(R, g)
        print(f'pool k={k} s={s}: max fd {fd(lambda: np.sum(R*maxpool_ff(X,g)[0]), X, DX, h=1e-5):.0e}, avg fd {fd(lambda: np.sum(R*avgpool_ff(X,g)), X, DXa, h=1e-5):.0e}')

    # flatten = im2col with a full-image kernel: gives (N, H*W*C) in NHWC order
    N, H, W, C = 2, 3, 4, 5; X = rng.standard_normal((N*H*W, C)); gf = G(N, H, W, H, W)
    F = im2col(X, gf); print('flatten shape', F.shape, 'equals NHWC flatten:', np.array_equal(F, X.reshape(N, H*W*C)),
                             ' unflatten via col2im:', np.array_equal(col2im(F, gf), X))

    # worked example for the specification: N=1, 3x3 image, 1 channel, 2x2 kernel
    g = G(1, 3, 3, 2, 2); X = np.arange(1, 10, dtype=float).reshape(9, 1)
    print('S_01 =\n', S(g, 0, 1).astype(int)); print('im2col(X) =\n', im2col(X, g).astype(int))
    print('col2im(ones) =', col2im(np.ones((4, 4)), g).ravel().astype(int))
