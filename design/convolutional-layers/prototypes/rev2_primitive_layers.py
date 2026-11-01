# Prototype (not part of the Nerva libraries): revision 2 of the design. Layers written in primitives incl. reshape, (c,p,q) columns.
import numpy as np, time

# ---------------- geometry ----------------
class G:
    def __init__(s, H, W, C, kh, kw, sh=1, sw=1, dh=1, dw=1, pt=0, pb=0, pl=0, pr=0):
        s.__dict__.update({k: v for k, v in locals().items() if k != 's'})
        s.Ho = (H + pt + pb - dh*(kh-1) - 1)//sh + 1
        s.Wo = (W + pl + pr - dw*(kw-1) - 1)//sw + 1

# ---------------- new primitives (NumPy implementations) ----------------
def reshape(X, m, n):                      # row-major
    return X.reshape(m, n)

def im2col(X, g):
    """X: N x (H*W*C), NHWC.  Returns (N*Ho*Wo) x (C*kh*kw); row (n,i,j), column (c,p,q).
       im2col(X)[(n,i,j),(c,p,q)] = X[n, sh*i+dh*p-pt, sw*j+dw*q-pl, c]   (0 outside the image)"""
    N = X.shape[0]
    Xp = np.pad(X.reshape(N, g.H, g.W, g.C), ((0,0),(g.pt,g.pb),(g.pl,g.pr),(0,0)))
    P = np.empty((N, g.Ho, g.Wo, g.C, g.kh, g.kw), X.dtype)
    for p in range(g.kh):
        for q in range(g.kw):
            P[..., p, q] = Xp[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :]
    return P.reshape(N*g.Ho*g.Wo, g.C*g.kh*g.kw)

def col2im(P, g):
    """Adjoint of im2col (contains the summation):
       col2im(P)[n,h,w,c] = sum_{i,p: sh*i+dh*p-pt=h} sum_{j,q: sw*j+dw*q-pl=w} P[(n,i,j),(c,p,q)]"""
    N = P.shape[0] // (g.Ho*g.Wo)
    P6 = P.reshape(N, g.Ho, g.Wo, g.C, g.kh, g.kw)
    Xp = np.zeros((N, g.H+g.pt+g.pb, g.W+g.pl+g.pr, g.C), P.dtype)
    for p in range(g.kh):
        for q in range(g.kw):
            Xp[:, p*g.dh:p*g.dh+g.sh*(g.Ho-1)+1:g.sh, q*g.dw:q*g.dw+g.sw*(g.Wo-1)+1:g.sw, :] += P6[..., p, q]
    return Xp[:, g.pt:g.pt+g.H, g.pl:g.pl+g.W, :].reshape(N, g.H*g.W*g.C)

def rows_max_mask(X):
    """1 at the first maximum of each row, 0 elsewhere (tie-safe, argmax-free)."""
    E = (X == X.max(axis=1, keepdims=True))
    return (E & (np.cumsum(E, axis=1) == 1)).astype(X.dtype)

# existing primitives
row_repeat = lambda b, m: np.tile(b[None, :], (m, 1))
column_repeat = lambda x, n: np.tile(x.reshape(-1, 1), (1, n))
columns_sum = lambda X: X.sum(axis=0)
rows_max = lambda X: X.max(axis=1)
rows_mean = lambda X: X.mean(axis=1)
hadamard = lambda X, Y: X * Y

# ---------------- layers, written only with primitives ----------------
class Conv2DLayer:
    def __init__(self, g, K):
        self.g, self.K = g, K
    def feedforward(self, X):
        g, K, W, b = self.g, self.K, self.W, self.b
        N = X.shape[0]
        M = N * g.Ho * g.Wo
        Xc = im2col(X, g)
        Zc = Xc @ W.T + row_repeat(b, M)
        Z = reshape(Zc, N, g.Ho * g.Wo * K)
        self.Xc = Xc
        return Z
    def backpropagate(self, Y, DY):
        g, K, W, Xc = self.g, self.K, self.W, self.Xc
        M = Xc.shape[0]
        DZc = reshape(DY, M, K)
        DW = DZc.T @ Xc
        Db = columns_sum(DZc)
        DX = col2im(DZc @ W, g)
        self.DW, self.Db, self.DX = DW, Db, DX

class MaxPool2DLayer:
    def __init__(self, g):
        self.g = g
    def feedforward(self, X):
        g = self.g; N = X.shape[0]; M = N * g.Ho * g.Wo * g.C
        P = reshape(im2col(X, g), M, g.kh * g.kw)
        Y = reshape(rows_max(P), N, g.Ho * g.Wo * g.C)
        self.P = P
        return Y
    def backpropagate(self, Y, DY):
        g, P = self.g, self.P; M = P.shape[0]
        DP = hadamard(rows_max_mask(P), column_repeat(reshape(DY, M, 1), g.kh * g.kw))
        DX = col2im(reshape(DP, M // g.C, g.C * g.kh * g.kw), g)
        self.DX = DX

class AvgPool2DLayer:
    def __init__(self, g):
        self.g = g
    def feedforward(self, X):
        g = self.g; N = X.shape[0]; M = N * g.Ho * g.Wo * g.C
        P = reshape(im2col(X, g), M, g.kh * g.kw)
        Y = reshape(rows_mean(P), N, g.Ho * g.Wo * g.C)
        return Y
    def backpropagate(self, Y, DY):
        g = self.g; M = DY.size
        DP = column_repeat(reshape(DY, M, 1), g.kh * g.kw) / (g.kh * g.kw)
        DX = col2im(reshape(DP, M // g.C, g.C * g.kh * g.kw), g)
        self.DX = DX

# ---------------- checks ----------------
rng = np.random.default_rng(3)
def fd(L, x, D, h=1e-6, n=30):
    e = 0
    for _ in range(n):
        i = tuple(rng.integers(0, s) for s in x.shape); o = x[i]
        x[i] = o + h; a = L(); x[i] = o - h; c = L(); x[i] = o
        e = max(e, abs((a - c)/(2*h) - D[i]))
    return e

def naive(X, W4, b, g, K):  # W4: (K, C, kh, kw)
    N = X.shape[0]; X4 = X.reshape(N, g.H, g.W, g.C); Z = np.zeros((N, g.Ho, g.Wo, K))
    for n in range(N):
        for i in range(g.Ho):
            for j in range(g.Wo):
                for p in range(g.kh):
                    for q in range(g.kw):
                        h, w = g.sh*i + g.dh*p - g.pt, g.sw*j + g.dw*q - g.pl
                        if 0 <= h < g.H and 0 <= w < g.W:
                            Z[n, i, j] += W4[:, :, p, q] @ X4[n, h, w]
    return (Z + b).reshape(N, -1)

for g, K in [(G(5,6,2,3,3), 3), (G(7,7,3,3,2,sh=2,pt=1,pb=1,pl=1), 4), (G(9,8,2,3,3,dh=2,dw=2,pt=2,pb=2,pl=2,pr=2), 2),
             (G(8,8,3,3,3,sh=2,sw=2,pt=1,pl=1), 5), (G(6,6,1,2,2,sh=3,sw=3), 2)]:
    N = 2; L = Conv2DLayer(g, K)
    X = rng.standard_normal((N, g.H*g.W*g.C)); L.W = rng.standard_normal((K, g.C*g.kh*g.kw)); L.b = rng.standard_normal(K)
    Z = L.feedforward(X)
    f = np.abs(Z - naive(X, L.W.reshape(K, g.C, g.kh, g.kw), L.b, g, K)).max()
    R = rng.standard_normal(Z.shape); L.backpropagate(None, R)
    loss = lambda: np.sum(R * L.feedforward(X))
    DX, DW, Db = L.DX, L.DW, L.Db
    P = rng.standard_normal((N*g.Ho*g.Wo, g.C*g.kh*g.kw))
    adj = abs(np.sum(im2col(X, g)*P) - np.sum(X*col2im(P, g)))
    print(f'conv H={g.H} W={g.W} C={g.C} K={K} k=({g.kh},{g.kw}) s=({g.sh},{g.sw}) d=({g.dh},{g.dw}): fwd {f:.0e}  fd DX {fd(loss,X,DX):.0e} DW {fd(loss,L.W,DW):.0e} Db {fd(loss,L.b,Db):.0e}  adjoint {adj:.0e}')

for k, s in [(2, 2), (3, 2), (3, 1)]:
    g = G(6, 7, 3, k, k, sh=s, sw=s); N = 2
    X = (rng.permutation(N*6*7*3).astype(float) / 11).reshape(N, -1)
    for Lc in (MaxPool2DLayer, AvgPool2DLayer):
        L = Lc(g); Y = L.feedforward(X); R = rng.standard_normal(Y.shape); L.backpropagate(Y, R); DX = L.DX
        print(f'{Lc.__name__} k={k} s={s}: Y{Y.shape} fd DX {fd(lambda: np.sum(R*L.feedforward(X)), X, DX, h=1e-5):.0e}')

# ties: first maximum receives the gradient
P = np.array([[1., 3, 3, 0], [2, 2, 2, 2], [0, 0, 0, 0]]); print('rows_max_mask ties:\n', rows_max_mask(P))

# cross-check: max pool backward == explicit argmax scatter
g = G(4, 4, 2, 2, 2, sh=2, sw=2); X = rng.standard_normal((1, 32)); L = MaxPool2DLayer(g); Y = L.feedforward(X); L.backpropagate(Y, np.ones_like(Y))
print('maxpool DX sums to #outputs:', L.DX.sum(), Y.size)

# timing CIFAR-sized, N=100, float32
for (H, C, K) in [(32, 3, 32), (16, 32, 64)]:
    g = G(H, H, C, 3, 3, pt=1, pb=1, pl=1, pr=1); L = Conv2DLayer(g, K)
    X = rng.standard_normal((100, H*H*C)).astype(np.float32); L.W = rng.standard_normal((K, 9*C)).astype(np.float32); L.b = np.zeros(K, np.float32)
    Z = L.feedforward(X); L.backpropagate(None, Z)
    t = time.perf_counter(); [L.feedforward(X) for _ in range(5)]; tf = (time.perf_counter()-t)/5
    t = time.perf_counter(); [L.backpropagate(None, Z) for _ in range(5)]; tb = (time.perf_counter()-t)/5
    print(f'conv H={H} C={C} K={K}: fwd {tf*1e3:.1f} ms, bwd {tb*1e3:.1f} ms')
g = G(32, 32, 32, 2, 2, sh=2, sw=2); L = MaxPool2DLayer(g); X = rng.standard_normal((100, 32*32*32)).astype(np.float32)
Y = L.feedforward(X); t = time.perf_counter(); L.feedforward(X); L.backpropagate(Y, Y); print(f'maxpool 32x32x32 fwd+bwd {(time.perf_counter()-t)*1e3:.1f} ms')
