// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_dense_matrix.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H
#define NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H

#include <mkl.h>
#include <stdexcept>
#include <vector>

namespace nerva::mkl {

// Uses the same values as Eigen
enum matrix_layout
{
  column_major = 0,
  row_major = 0x1
};

// This class can be used to wrap an Eigen matrix (or NumPy etc.)
template <typename Scalar, int MatrixLayout>
class dense_matrix_view
{
  protected:
    Scalar* values;
    long m; // number of rows
    long n; // number of columns

  public:
    dense_matrix_view(Scalar* values_, long m_, long n_)
      : values(values_), m(m_), n(n_)
    {}

    [[nodiscard]] long rows() const
    {
      return m;
    }

    [[nodiscard]] long cols() const
    {
      return n;
    }

    Scalar* data()
    {
      return values;
    }

    const Scalar* data() const
    {
      return values;
    }
};

template <typename Scalar, int MatrixLayout>
class dense_matrix
{
  protected:
    std::vector<Scalar> values;
    long m; // number of rows
    long n; // number of columns

  public:
    dense_matrix(long m_, long n_)
      : values(m_ * n_, 0), m(m_), n(n_)
    {}

    [[nodiscard]] long rows() const
    {
      return m;
    }

    [[nodiscard]] long cols() const
    {
      return n;
    }

    Scalar* data()
    {
      return values.data();
    }

    const Scalar* data() const
    {
      return values.data();
    }
};

// Computes the matrix product C = A * B
template <typename Scalar, int MatrixLayout, template <typename, int> class Matrix1, template <typename, int> class Matrix2>
dense_matrix<Scalar, MatrixLayout> matrix_product(const Matrix1<Scalar, MatrixLayout>& A, const Matrix2<Scalar, MatrixLayout>& B)
{
  assert(A.cols() == B.rows());
  dense_matrix<Scalar, MatrixLayout> C(A.rows(), B.cols());

  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;

  if constexpr (MatrixLayout == column_major)
  {
    int lda = A.rows();
    int ldb = B.rows();
    int ldc = C.rows();
    double alpha = 1.0;
    double beta = 0.0;
    if constexpr (std::is_same<Scalar, double>::value)
    {
      cblas_dgemm(CblasColMajor, transA, transB, C.rows(), C.cols(), A.cols(), alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    else
    {
      cblas_sgemm(CblasColMajor, transA, transB, C.rows(), C.cols(), A.cols(), alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
  }
  else
  {
    int lda = A.cols();
    int ldb = B.cols();
    int ldc = C.cols();
    double alpha = 1.0;
    double beta = 0.0;
    if constexpr (std::is_same<Scalar, double>::value)
    {
      cblas_dgemm(CblasRowMajor, transA, transB, C.rows(), C.cols(), A.cols(), alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    else
    {
      cblas_sgemm(CblasRowMajor, transA, transB, C.rows(), C.cols(), A.cols(), alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
  }
  return C;
}

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H
