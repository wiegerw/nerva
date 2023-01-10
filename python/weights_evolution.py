# This code is from https://github.com/Shiweiliuiiiiiii/SET-MLP-ONE-MILLION-NEURONS/blob/master/set_mlp_sparse_data_structures.py

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix


def create_sparse_weights(epsilon, noRows, noCols, gen = lambda: np.random.randn() / 10):
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float64(gen())
    print("Created sparse matrix with ", weights.getnnz(), " connections and ", (weights.getnnz() / (noRows * noCols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def weights_evolution_I(w: csr_matrix, zeta: float, gen = lambda: np.random.randn() / 10):
    # It removes the weights closest to zero in each layer and add new random weights
    rows, columns = w.shape

    values = np.sort(w.data)
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)

    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[
        int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]

    wlil = w.tolil()
    wdok = dok_matrix((rows, columns), dtype="float64")

    # remove the weights closest to zero
    keepConnections = 0
    for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
        for jk, val in zip(row, data):
            if ((val < largestNegative) or (val > smallestPositive)):
                wdok[ik, jk] = val
                keepConnections += 1

    # add new random connections
    for kk in range(w.data.shape[0] - keepConnections):
        ik = np.random.randint(0, rows)
        jk = np.random.randint(0, columns)
        while (wdok[ik, jk] != 0):
            ik = np.random.randint(0, rows)
            jk = np.random.randint(0, columns)
        wdok[ik, jk] = gen()

    return wdok.tocsr()


def weights_evolution_II(w: csr_matrix, zeta: float, gen = lambda: np.random.randn() / 10):

    def array_intersect(A, B):
        # inspired by https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
        nrows, ncols = A.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
        return np.in1d(A.view(dtype), B.view(dtype))  # boolean return

    # It removes the weights closest to zero in each layer and add new random weights
    rows, columns = w.shape

    wcoo = w.tocoo()
    valsW = wcoo.data
    rowsW = wcoo.row
    colsW = wcoo.col

    # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
    values = np.sort(w.data)
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)

    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]

    # remove the weights (W) closest to zero and modify PD as well
    valsWNew = valsW[(valsW > smallestPositive) | (valsW < largestNegative)]
    rowsWNew = rowsW[(valsW > smallestPositive) | (valsW < largestNegative)]
    colsWNew = colsW[(valsW > smallestPositive) | (valsW < largestNegative)]

    # add new random connections
    keepConnections = np.size(rowsWNew)
    lengthRandom = valsW.shape[0] - keepConnections
    randomVals = np.empty(lengthRandom)
    for i in range(lengthRandom):
        randomVals[i] = gen()

    # adding  (wdok[ik,jk]!=0): condition
    while (lengthRandom > 0):
        ik = np.random.randint(0, rows, size=lengthRandom, dtype='int32')
        jk = np.random.randint(0, columns, size=lengthRandom, dtype='int32')

        randomWRowColIndex = np.stack((ik, jk), axis=-1)
        randomWRowColIndex = np.unique(randomWRowColIndex, axis=0)  # removing duplicates in new rows&cols
        oldWRowColIndex = np.stack((rowsWNew, colsWNew), axis=-1)

        uniqueFlag = ~array_intersect(randomWRowColIndex, oldWRowColIndex)  # careful about order & tilda

        ikNew = randomWRowColIndex[uniqueFlag][:, 0]
        jkNew = randomWRowColIndex[uniqueFlag][:, 1]
        # be careful - row size and col size needs to be verified
        rowsWNew = np.append(rowsWNew, ikNew)
        colsWNew = np.append(colsWNew, jkNew)

        lengthRandom = valsW.shape[0] - np.size(rowsWNew)  # this will constantly reduce lengthRandom

    # adding all the values along with corresponding row and column indices - Added by Amar
    valsWNew = np.append(valsWNew, randomVals)  # be careful - we can add to an existing link ?

    if (valsWNew.shape[0] != rowsWNew.shape[0]):
        print("not good")

    return coo_matrix((valsWNew, (rowsWNew, colsWNew)), (rows, columns)).tocsr()

    # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(w.data.shape[0]), (self.pdw[i].data.shape[0])])
