#!/usr/bin/env python3

CODE = r'''#------------------------------------------#
#          simple_batch_normalization_layer_colwise
#------------------------------------------#

R = X - column_repeat(rows_mean(X), N)
Sigma = diag(R * R.T) / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Y = hadamard(column_repeat(inv_sqrt_Sigma, N), R)

DX = hadamard(column_repeat(inv_sqrt_Sigma / N, N), hadamard(Y, column_repeat(-diag(DY * Y.T), N)) + DY * (N * identity(N) - ones(N, N)))

#------------------------------------------#
#          affine_layer_colwise
#------------------------------------------#

Y = hadamard(column_repeat(gamma, N), X) + column_repeat(beta, N)

DX = hadamard(column_repeat(gamma, N), DY)
Dbeta = rows_sum(DY)
Dgamma = rows_sum(hadamard(X, DY))

#------------------------------------------#
#          batch_normalization_layer_colwise
#------------------------------------------#

R = X - column_repeat(rows_mean(X), N)
Sigma = diag(R * R.T) / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Z = hadamard(column_repeat(inv_sqrt_Sigma, N), R)
Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

DZ = hadamard(column_repeat(gamma, N), DY)
Dbeta = rows_sum(DY)
Dgamma = rows_sum(hadamard(DY, Z))
DX = hadamard(column_repeat(inv_sqrt_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

#------------------------------------------#
#          simple_batch_normalization_layer_rowwise
#------------------------------------------#

R = X - row_repeat(columns_mean(X), N)
Sigma = diag(R.T * R).T / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Y = hadamard(row_repeat(inv_sqrt_Sigma, N), R)

DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) * DY - hadamard(Y, row_repeat(diag(Y.T * DY).T, N)))

#------------------------------------------#
#          affine_layer_rowwise
#------------------------------------------#

Y = hadamard(row_repeat(gamma, N), X) + row_repeat(beta, N)

DX = hadamard(row_repeat(gamma, N), DY)
Dbeta = columns_sum(DY)
Dgamma = columns_sum(hadamard(X, DY))

#------------------------------------------#
#          batch_normalization_layer_rowwise
#------------------------------------------#

R = X - row_repeat(columns_mean(X), N)
Sigma = diag(R.T * R).T / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

DZ = hadamard(row_repeat(gamma, N), DY)
Dbeta = columns_sum(DY)
Dgamma = columns_sum(hadamard(DY, Z))
DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, row_repeat(diag(Z.T * DZ).T, N)))

#------------------------------------------#
#          yeh_batch_normalization_layer_rowwise
#------------------------------------------#

R = X - row_repeat(columns_mean(X), N)
Sigma = diag(R.T * R).T / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

DZ = hadamard(row_repeat(gamma, N), DY)  # this equation is not explicitly given in [Yeh 2017]
Dbeta = columns_sum(DY)                  # this equation is the same as in [Yeh 2017]
Dgamma = columns_sum(hadamard(DY, Z))    # I can't parse the equation in [Yeh 2017], but this is probably it
DX = (1 / N) * (-hadamard(row_repeat(Dgamma, N), Z) + N * DY - row_repeat(Dbeta, N)) * row_repeat(hadamard(gamma, Sigma), D) # I can't parse the equation in [Yeh 2017], but this is probably it

#------------------------------------------#
#          linear_dropout_layer_colwise
#------------------------------------------#

Y = hadamard(W, R) * X + column_repeat(b, N)

DW = hadamard(DY * X.T, R)
Db = rows_sum(DY)
DX = hadamard(W, R).T * DY

#------------------------------------------#
#          activation_dropout_layer_colwise
#------------------------------------------#

Z = hadamard(W, R) * X + column_repeat(b, N)
Y = act(Z)

DZ = hadamard(DY, act_gradient(Z))
DW = hadamard(DZ * X.T, R)
Db = rows_sum(DZ)
DX = hadamard(W, R).T * DZ

#------------------------------------------#
#          sigmoid_dropout_layer_colwise
#------------------------------------------#

Z = hadamard(W, R) * X + column_repeat(b, N)
Y = Sigmoid(Z)

DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
DW = hadamard(DZ * X.T, R)
Db = rows_sum(DZ)
DX = hadamard(W, R).T * DZ

#------------------------------------------#
#          linear_dropout_layer_rowwise
#------------------------------------------#

Y = X * hadamard(W.T, R) + row_repeat(b, N)

DW = hadamard(DY.T * X, R.T)
Db = columns_sum(DY)
DX = DY * hadamard(W, R.T)

#------------------------------------------#
#          activation_dropout_layer_rowwise
#------------------------------------------#

Z = X * hadamard(W.T, R) + row_repeat(b, N)
Y = act(Z)

DZ = hadamard(DY, act_gradient(Z))
DW = hadamard(DZ.T * X, R.T)
Db = columns_sum(DZ)
DX = DZ * hadamard(W, R.T)

#------------------------------------------#
#          sigmoid_dropout_layer_rowwise
#------------------------------------------#

Z = X * hadamard(W.T, R) + row_repeat(b, N)
Y = Sigmoid(Z)

DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
DW = hadamard(DZ.T * X, R.T)
Db = columns_sum(DZ)
DX = DZ * hadamard(W, R.T)

#------------------------------------------#
#          linear_layer_colwise
#------------------------------------------#

Y = W * X + column_repeat(b, N)

DW = DY * X.T
Db = rows_sum(DY)
DX = W.T * DY

#------------------------------------------#
#          activation_layer_colwise
#------------------------------------------#

Z = W * X + column_repeat(b, N)
Y = act(Z)

DZ = hadamard(DY, act.gradient(Z))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ

#------------------------------------------#
#          sigmoid_layer_colwise
#------------------------------------------#

Z = W * X + column_repeat(b, N)
Y = Sigmoid(Z)

DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ

#------------------------------------------#
#          linear_layer_rowwise
#------------------------------------------#

Y = X * W.T + row_repeat(b, N)

DW = DY.T * X
Db = columns_sum(DY)
DX = DY * W

#------------------------------------------#
#          activation_layer_rowwise
#------------------------------------------#

Z = X * W.T + row_repeat(b, N)
Y = act(Z)

DZ = hadamard(DY, act.gradient(Z))
DW = DZ.T * X
Db = columns_sum(DZ)
DX = DZ * W

#------------------------------------------#
#          sigmoid_layer_rowwise
#------------------------------------------#

Z = X * W.T + row_repeat(b, N)
Y = Sigmoid(Z)

DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
DW = DZ.T * X
Db = columns_sum(DZ)
DX = DZ * W

#------------------------------------------#
#          softmax_layer_colwise
#------------------------------------------#

Z = W * X + column_repeat(b, N)
Y = softmax_colwise(Z)

DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ

#------------------------------------------#
#          log_softmax_layer_colwise
#------------------------------------------#

Z = W * X + column_repeat(b, N)
Y = log_softmax_colwise(Z)

DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ

#------------------------------------------#
#          softmax_layer_rowwise
#------------------------------------------#

Z = X * W.T + row_repeat(b, N)
Y = softmax_rowwise(Z)

DZ = hadamard(Y, DY - column_repeat(diag(DY * Y.T), N))
DW = DZ.T * X
Db = columns_sum(DZ)
DX = DZ * W

#------------------------------------------#
#          log_softmax_layer_rowwise
#------------------------------------------#

Z = X * W.T + row_repeat(b, N)
Y = log_softmax_rowwise(Z)

DZ = DY - hadamard(softmax_rowwise(Z), column_repeat(rows_sum(DY), N))
DW = DZ.T * X
Db = columns_sum(DZ)
DX = DZ * W

#------------------------------------------#
#          srelu_layer_colwise
#------------------------------------------#

Z = W * X + column_repeat(b, N)
Y = act(Z)

DZ = hadamard(DY, act.gradient(Z))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ
Dal = elements_sum(hadamard(DY, Al(Z)))
Dar = elements_sum(hadamard(DY, Ar(Z)))
Dtl = elements_sum(hadamard(DY, Tl(Z)))
Dtr = elements_sum(hadamard(DY, Tr(Z)))

#------------------------------------------#
#          srelu_layer_rowwise
#------------------------------------------#

Z = X * W.T + row_repeat(b, N)
Y = act(Z)

DZ = hadamard(DY, act.gradient(Z))
DW = DZ.T * X
Db = columns_sum(DZ)
DX = DZ * W
Dal = elements_sum(hadamard(DY, Al(Z)))
Dar = elements_sum(hadamard(DY, Ar(Z)))
Dtl = elements_sum(hadamard(DY, Tl(Z)))
Dtr = elements_sum(hadamard(DY, Tr(Z)))'''

def split_paragraphs(text: str) -> list[str]:
    return text.strip().split('\n\n')


def make_equations(equations, text):
    paragraphs = split_paragraphs(equations.strip())
    if len(paragraphs) == 5:
        paragraphs.append('')
    header, fftext, bptext, ffcode, bpcode, footer = paragraphs
    text = text.replace('AAA', fftext.strip())
    text = text.replace('BBB', bptext.strip())
    text = text.replace('CCC', ffcode.strip())
    text = text.replace('DDD', bpcode.strip())
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(header)
    print(text)
    if footer:
        print(footer)
    print('\n')


def run():
    paragraphs = split_paragraphs(CODE)
    for header, feedforward, backpropagate in zip(paragraphs[0::3], paragraphs[1::3], paragraphs[2::3]):
        print(f'''{header}
\\\\
\\addlinespace[2ex]
\\begin{{minipage}}[t]{{0.5\\textwidth}}
\\small\\begin{{verbatim}}
{feedforward}
\\end{{verbatim}}
\\end{{minipage}}
&
\\begin{{minipage}}[t]{{0.5\\textwidth}}
\\small\\begin{{verbatim}}
{backpropagate}
\\end{{verbatim}}
\\end{{minipage}} \\\\
\\end{{array}}
\\]
''')

run()
