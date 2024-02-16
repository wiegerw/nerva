#!/usr/bin/env python3

TEXT1 = r'''\[
\begin{array}{@{} *{2}{L} @{}}
\textsc{feedforward equations} & \textsc{backpropagation equations} \\
\addlinespace[1ex]
\begin{aligned}[t]
AAA
\end{aligned}
&
\begin{aligned}[t]
BBB
\end{aligned}
\\
\addlinespace[2ex]
\begin{minipage}[t]{0.5\textwidth}
\begin{minted}[fontsize=\small]{python}
CCC
\end{minted}
\end{minipage}
&
\begin{minipage}[t]{0.5\textwidth}
\begin{minted}[fontsize=\small]{python}
DDD
\end{minted}
\end{minipage} \\
\end{array}
\]'''

LINEAR_LAYER_TEXT = r'''
\subsection*{linear-layer}

Y &= X W^\top + 1_N \cdot b

\Derivative W &= \Derivative Y^\top \cdot X
\\
\Derivative b &= 1_N^\top \cdot \Derivative Y
\\
\Derivative X &= \Derivative Y \cdot W

Y = X * W.T + repeat_row(b, N)

DW = DY.T * X
Db = sum_columns(DY)
DX = DY * W
'''

ACTIVATION_LAYER_TEXT = r'''
\subsection*{activation-layer}
Let $\func{act}: \mathbb{R} \rightarrow \mathbb{R}$ be an activation function, for example \func{relu}.

Z &= X W^\top + 1_N \cdot b \\
Y &= \func{act}(Z)

\Derivative W &= \Derivative Z^\top \cdot X
\\
\Derivative b &= 1_N^\top \cdot \Derivative Z
\\
\Derivative X &= \Derivative Z \cdot W

Z = X * W.T + repeat_row(b, N)
Y = apply(act, Z)

DZ = hadamard(DY, apply(act_prime, Z))
DW = DZ.T * X
Db = sum_columns(DZ)
DX = DZ * W
'''

SIGMOID_LAYER_TEXT = r'''
\subsection*{sigmoid-layer}

Z &= X W^\top + 1_N \cdot b
\\
Y &= \sigma(Z)

\Derivative Z &= \Derivative Y \odot \sigma'(Z) = \Derivative Y \odot Y \odot (1_{NK} - Y)
\\
\Derivative W &= \Derivative Z^\top \cdot X
\\
\Derivative b &= 1_N^\top \cdot \Derivative Z
\\
\Derivative X &= \Derivative Z \cdot W

Z = X * W.T + repeat_row(b, N)
Y = apply(sigma, Z)

DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
DW = DZ.T * X
Db = sum_columns(DZ)
DX = DZ * W
'''

SRELU_LAYER_TEXT = r'''
\subsection*{srelu-layer}

Z &= X W^\top + 1_N \cdot b
\\
Y &= \func{srelu}(Z)

\Derivative Z &= \Derivative Y \odot \func{srelu}'(Z)
\\
\Derivative W &= \Derivative Z^\top \cdot X
\\
\Derivative b &= 1_N^\top \cdot \Derivative Z
\\
\Derivative X &= \Derivative Z \cdot W
\\
\Derivative a_l &= 1_N^\top \cdot (\Derivative Y \odot A^l) \cdot 1_K
\\
\Derivative a_r &= 1_N^\top \cdot (\Derivative Y \odot A^r) \cdot 1_K
\\
\Derivative t_l &= 1_N^\top \cdot (\Derivative Y \odot T^l) \cdot 1_K
\\
\Derivative t_r &= 1_N^\top \cdot (\Derivative Y \odot T^r) \cdot 1_K

Z = X * W.T + repeat_row(b, N)
Y = apply(act, Z)

DZ = hadamard(DY, apply(act_prime, Z))
DW = DZ.T * X
Db = sum_columns(DZ)
DX = DZ * W
Zij = sp.symbols('Zij')
Al = apply(..., Z)  # implementation details omitted
Ar = apply(..., Z)  # implementation details omitted
Tl = apply(..., Z)  # implementation details omitted
Tr = apply(..., Z)  # implementation details omitted
Dal = sum_elements(hadamard(DY, Al))
Dar = sum_elements(hadamard(DY, Ar))
Dtl = sum_elements(hadamard(DY, Tl))
Dtr = sum_elements(hadamard(DY, Tr))

where
\[
    \begin{array}{lll}
      A^l_{ij} &=&
          \begin{cases*}
              Z_{ij} - t_l & if $Z_{ij} \leq t_l$ \\
              0 & otherwise
          \end{cases*}
      \\[0.5cm]
      A^r_{ij} &=&
          \begin{cases*}
              0 & if $Z_{ij} \leq t_l \lor Z_{ij} < t_r$ \\
              Z_{ij} - t_r & otherwise
          \end{cases*}
      \\[0.5cm]
      T^l_{ij} &=&
          \begin{cases*}
              1 - a_l & if $Z_{ij} \leq t_l$ \\
              0 & otherwise
          \end{cases*}
      \\[0.5cm]
      T^r_{ij} &=&
          \begin{cases*}
              1 - a_r & if $Z_{ij} \geq t_r$ \\
              0 & otherwise
          \end{cases*}
    \end{array}
\]
'''

SOFTMAX_LAYER_TEXT = r'''
\subsection*{softmax-layer}

Z &= X W^\top + 1_N \cdot b
\\
Y &= \func{softmax}(Z)

\Derivative Z &= Y \odot (\Derivative Y - 1_K \cdot \func{diag}(Y^\top \Derivative Y)^\top)
\\
\Derivative W &= \Derivative Z \cdot X^\top
\\
\Derivative b &= \Derivative Z \cdot 1_N
\\
\Derivative X &= W^T \Derivative Z

Z = X * W.T + repeat_row(b, N)
Y = softmax_rowwise(Z)

DZ = hadamard(Y, DY - repeat_column(diag(DY * Y.T), N))
DW = DZ.T * X
Db = sum_columns(DZ)
DX = DZ * W
'''

LOG_SOFTMAX_LAYER_TEXT = r'''
\subsection*{log-softmax-layer}

Z &= X W^\top + 1_N \cdot b
\\
Y &= \func{logsoftmax}(Z)

\Derivative Z &= \Derivative Y - \func{softmax}(Z) \cdot 1_K \cdot 1_K^\top \cdot \Derivative Y
\\
\Derivative W &= \Derivative Z \cdot X^\top
\\
\Derivative b &= \Derivative Z \cdot 1_N
\\
\Derivative X &= W^T \Derivative Z

Z = X * W.T + repeat_row(b, N)
Y = log_softmax_rowwise(Z)

DZ = DY - hadamard(softmax_rowwise(Z),
                   repeat_column(sum_rows(DY), N))
DW = DZ.T * X
Db = sum_columns(DZ)
DX = DZ * W
'''

SIMPLE_BATCH_NORMALIZATION_LAYER_TEXT = r'''
\subsection*{simple-batch-normalization-layer}

R &= (\mathbb{I}_{N} - \frac{1_N \cdot 1_N^\top}{N}) \cdot X
\\
\Sigma &= \frac{1}{N} \cdot \func{diag}(R^\top R)^\top
\\
Y &= (1_N \cdot \Sigma^{-\frac{1}{2}}) \odot R

\Derivative X
     &= (\frac{1}{N} \cdot 1_N \cdot \Sigma^{-\frac{1}{2}}) ~\odot \\
     & \left(
              (N \cdot \mathbb{I}_N - 1_N \cdot 1_N^\top) \cdot \Derivative Y
              -
              Y \odot (1_N \cdot \func{diag}(Y^\top \cdot \Derivative Y)^\top)
       \right)

R = (identity(N) - ones(N, N) / N) * X
Sigma = diag(R.T * R).T / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Y = hadamard(repeat_row(inv_sqrt_Sigma, N), R)

DX = hadamard(repeat_row(inv_sqrt_Sigma / N, N),
     (N * identity(N) - ones(N, N)) * DY -
     hadamard(Y, repeat_row(diag(Y.T * DY).T, N)))
'''

AFFINE_LAYER_TEXT = r'''
\subsection*{affine-layer}

Y &= (1_N \cdot \gamma) \odot X + 1_N \cdot \beta

\Derivative X &= (1_N \cdot \gamma) \odot \Derivative Y
\\
\Derivative \beta &= 1_N^\top \cdot \Derivative Y
\\
\Derivative \gamma &= 1_N^\top \cdot (\Derivative Y \odot X)

Y = hadamard(repeat_row(gamma, N), X) +
             repeat_row(beta, N)

DX = hadamard(repeat_row(gamma, N), DY)
Dbeta = sum_columns(DY)
Dgamma = sum_columns(hadamard(X, DY))
'''

BATCH_NORMALIZATION_LAYER_TEXT = r'''
\subsection*{batch-normalization-layer}
Note that the equations of a batch normalization layer can be obtained by combining
the equations of a simple batch normalization layer and an affine layer.

R &= (\mathbb{I}_{N} - \frac{1_N \cdot 1_N^\top}{N}) \cdot X
\\
\Sigma &= \frac{1}{N} \cdot \func{diag}(R^\top R)^\top
\\
Z &= (1_N \cdot \Sigma^{-\frac{1}{2}}) \odot R
\\
Y &= (1_N \cdot \gamma) \odot Z + 1_N \cdot \beta

\Derivative Z &= (1_N \cdot \gamma) \odot \Derivative Y
\\
\Derivative \beta &= 1_N^\top \cdot \Derivative Y
\\
\Derivative \gamma &= 1_N^\top \cdot (\Derivative Y \odot Z)
\\
\Derivative X
     &= (\frac{1}{N} \cdot 1_N \cdot \Sigma^{-\frac{1}{2}}) ~\odot \\
     &  \left(
              (N \cdot \mathbb{I}_N - 1_N \cdot 1_N^\top) \cdot \Derivative Z
              -
              Z \odot (1_N \cdot \func{diag}(Z^\top \cdot \Derivative Z)^\top)
       \right)

R = (identity(N) - ones(N, N) / N) * X
Sigma = diag(R.T * R).T / N
inv_sqrt_Sigma = inv_sqrt(Sigma)
Z = hadamard(repeat_row(
             inv_sqrt_Sigma, N), R)
Y = hadamard(repeat_row(gamma, N), Z) +
             repeat_row(beta, N)

DZ = hadamard(repeat_row(gamma, N), DY)
Dbeta = sum_columns(DY)
Dgamma = sum_columns(hadamard(Z, DY))
DX = hadamard(repeat_row(inv_sqrt_Sigma / N, N),
     (N * identity(N) - ones(N, N)) * DZ -
     hadamard(Z, repeat_row(diag(Z.T * DZ).T, N)))
'''

LINEAR_DROPOUT_LAYER_TEXT = r'''
\subsection*{linear-dropout-layer}

Y &= (W \odot R) X + b \cdot 1_N^\top

\Derivative W &= (\Derivative Y \cdot X^\top) \odot R
\\
\Derivative b &= \Derivative Y \cdot 1_N
\\
\Derivative X &= (W \odot R)^\top \Derivative Y

Y = X * hadamard(W, R).T + row_repeat(b, N)

DW = hadamard(DY.T * X, R)
Db = columns_sum(DY)
DX = DY * hadamard(W, R)
'''

ACTIVATION_DROPOUT_LAYER_TEXT = r'''
\subsection*{activation-dropout-layer}

Z &= (W \odot R) X + b \cdot 1_N^\top
\\
Y &= \func{act}(Z)

\Derivative Z &= \Derivative Y \odot \func{act}'(Z)
\\
\Derivative W &= (\Derivative Z \cdot X^\top) \odot R
\\
\Derivative b &= \Derivative Z \cdot 1_N
\\
\Derivative X &= (W \odot R)^\top \Derivative Z

Z = X * hadamard(W, R).T + row_repeat(b, N)
Y = apply(act, Z)

DZ = hadamard(DY, apply(act_prime, Z))
DW = hadamard(DZ.T * X, R)
Db = columns_sum(DZ)
DX = DZ * hadamard(W, R)
'''

SIGMOID_DROPOUT_LAYER_TEXT = r'''
\subsection*{sigmoid-dropout-layer}
In case of the sigmoid activation function, the backpropagate function can be rewritten, which might be slightly more efficient.

Z &= (W \odot R) X + b \cdot 1_N^\top
\\
Y &= \sigma(Z)

\Derivative Z &= \Derivative Y \odot \sigma'(Z)
 = \Derivative Y \odot Y \odot (1_{KN} - Y)
\\
\Derivative W &= (\Derivative Z \cdot X^\top) \odot R
\\
\Derivative b &= \Derivative Z \cdot 1_N
\\
\Derivative X &= (W \odot R)^\top \Derivative Z

Z = X * hadamard(W, R).T + row_repeat(b, N)
Y = apply(sigma, Z)

DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
DW = hadamard(DZ.T * X, R)
Db = columns_sum(DZ)
DX = DZ * hadamard(W, R)
'''

def split_paragraphs(text: str) -> list[str]:
    return text.strip().split('\n\n')


def make_equations(equations, text=TEXT1):
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
    make_equations(LINEAR_LAYER_TEXT)
    make_equations(ACTIVATION_LAYER_TEXT)
    make_equations(SIGMOID_LAYER_TEXT)
    make_equations(SRELU_LAYER_TEXT)
    make_equations(SOFTMAX_LAYER_TEXT)
    make_equations(LOG_SOFTMAX_LAYER_TEXT)
    make_equations(SIMPLE_BATCH_NORMALIZATION_LAYER_TEXT)
    make_equations(AFFINE_LAYER_TEXT)
    make_equations(BATCH_NORMALIZATION_LAYER_TEXT)
    make_equations(LINEAR_DROPOUT_LAYER_TEXT)
    make_equations(ACTIVATION_DROPOUT_LAYER_TEXT)
    make_equations(SIGMOID_DROPOUT_LAYER_TEXT)

run()
