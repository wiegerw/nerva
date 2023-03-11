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
#include <cassert>
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
template <typename Scalar>
class dense_matrix_view
{
  protected:
    Scalar* m_data;
    long m_rows;
    long m_columns;
    int m_layout;

  public:
    dense_matrix_view(Scalar* data, long rows, long columns, int layout)
      : m_data(data), m_rows(rows), m_columns(columns), m_layout(layout)
    {}

    [[nodiscard]] long rows() const
    {
      return m_rows;
    }

    [[nodiscard]] long cols() const
    {
      return m_columns;
    }

    Scalar* data()
    {
      return m_data;
    }

    const Scalar* data() const
    {
      return m_data;
    }

    [[nodiscard]] int layout() const
    {
      return m_layout;
    }
};

template <typename Scalar>
class dense_matrix
{
  protected:
    std::vector<Scalar> m_data;
    long m_rows;
    long m_columns;
    int m_layout;  // TODO: use the type matrix_layout

  public:
    dense_matrix(long rows, long columns, int layout, bool transposed = false)
      : m_data(rows * columns, 0), m_rows(rows), m_columns(columns), m_layout(layout)
    {}

    [[nodiscard]] long rows() const
    {
      return m_rows;
    }

    [[nodiscard]] long cols() const
    {
      return m_columns;
    }

    Scalar* data()
    {
      return m_data.data();
    }

    const Scalar* data() const
    {
      return m_data.data();
    }

    [[nodiscard]] int layout() const
    {
      return m_layout;
    }
};

// Computes the matrix product C = A * B
template <typename Scalar, template <typename> class Matrix1, template <typename> class Matrix2>
dense_matrix<Scalar> ddd_product(const Matrix1<Scalar>& A, const Matrix2<Scalar>& B, bool A_transposed = false, bool B_transposed = false)
{
  long A_cols = A_transposed ? A.rows() : A.cols();
  long A_rows = A_transposed ? A.cols() : A.rows();
  long B_cols = B_transposed ? B.rows() : B.cols();
  long B_rows = B_transposed ? B.cols() : B.rows();

  assert(A.layout() == B.layout());
  assert(A_rows == B_cols);

  dense_matrix<Scalar> C(A_rows, B_cols, A.layout());

  CBLAS_TRANSPOSE transA = A_transposed ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = B_transposed ? CblasTrans : CblasNoTrans;

  if (A.layout() == column_major)
  {
    long lda = A_transposed ? A_cols : A_rows;
    long ldb = B_transposed ? B_cols : B_rows;
    long ldc = C.rows();
    double alpha = 1.0;
    double beta = 0.0;
    if constexpr (std::is_same<Scalar, double>::value)
    {
      cblas_dgemm(CblasColMajor, transA, transB, C.rows(), C.cols(), A_cols, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    else
    {
      cblas_sgemm(CblasColMajor, transA, transB, C.rows(), C.cols(), A_cols, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
  }
  else
  {
    long lda = A_transposed ? A_rows : A_cols;
    long ldb = B_transposed ? B_rows : B_cols;
    long ldc = C.cols();
    double alpha = 1.0;
    double beta = 0.0;
    if constexpr (std::is_same<Scalar, double>::value)
    {
      cblas_dgemm(CblasRowMajor, transA, transB, C.rows(), C.cols(), A_cols, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    else
    {
      cblas_sgemm(CblasRowMajor, transA, transB, C.rows(), C.cols(), A_cols, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
  }
  return C;
}

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H
