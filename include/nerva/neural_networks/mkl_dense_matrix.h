// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_dense_matrix.h
/// \brief add your file description here.

#pragma once

#include "Eigen/Dense"
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

inline
long column_major_index(long rows, [[maybe_unused]] long columns, long i, long j)
{
  assert(0 <= i && i < rows);
  assert(0 <= j && j < columns);
  return j * rows + i;
}

inline
long row_major_index([[maybe_unused]] long rows, long columns, long i, long j)
{
  assert(0 <= i && i < rows);
  assert(0 <= j && j < columns);
  return i * columns + j;
}

// This class can be used to wrap an Eigen matrix (or NumPy etc.)
template <typename Scalar, int MatrixLayout>
class dense_matrix_view
{
  protected:
    Scalar* m_data;
    long m_rows;
    long m_columns;

  public:
    dense_matrix_view(Scalar* data, long rows, long columns)
      : m_data(data), m_rows(rows), m_columns(columns)
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

    Scalar operator()(long i, long j) const
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_data[column_major_index(m_rows, m_columns, i, j)];
      }
      else
      {
        return m_data[row_major_index(m_rows, m_columns, i, j)];
      }
    }

    Scalar& operator()(long i, long j)
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_data[column_major_index(m_rows, m_columns, i, j)];
      }
      else
      {
        return m_data[row_major_index(m_rows, m_columns, i, j)];
      }
    }
};

template <typename Scalar, int MatrixLayout>
class dense_matrix
{
  protected:
    std::vector<Scalar> m_data;
    long m_rows;
    long m_columns;

  public:
    dense_matrix(long rows, long columns)
      : m_data(rows * columns, 0), m_rows(rows), m_columns(columns)
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

    Scalar operator()(long i, long j) const
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_data[column_major_index(m_rows, m_columns, i, j)];
      }
      else
      {
        return m_data[row_major_index(m_rows, m_columns, i, j)];
      }
    }

    Scalar& operator()(long i, long j)
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_data[column_major_index(m_rows, m_columns, i, j)];
      }
      else
      {
        return m_data[row_major_index(m_rows, m_columns, i, j)];
      }
    }
};

template <typename Scalar, int MatrixLayout, template <typename, int> class Matrix>
dense_matrix_view<Scalar, MatrixLayout> make_dense_matrix_view(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  return dense_matrix_view<Scalar, MatrixLayout>(const_cast<Scalar*>(A.data()), A.cols(), A.rows());
}

template <typename Derived>
auto make_dense_matrix_view(const Eigen::MatrixBase<Derived>& A)
{
  constexpr int MatrixLayout = Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;
  using Scalar = typename Derived::Scalar;
  return dense_matrix_view<Scalar, MatrixLayout>(const_cast<Scalar*>(A.derived().data()), A.rows(), A.cols());
}

template <typename Scalar, int MatrixLayout, template <typename, int> class Matrix>
dense_matrix_view<Scalar, 1 - MatrixLayout> make_transposed_dense_matrix_view(const Matrix<Scalar, MatrixLayout>& A)
{
  return dense_matrix_view<Scalar, 1 - MatrixLayout>(const_cast<Scalar*>(A.data()), A.cols(), A.rows());
}

// Computes the matrix product C = A * B
template <typename Scalar, int MatrixLayoutA, template <typename, int> class MatrixA, int MatrixLayoutB, template <typename, int> class MatrixB>
auto ddd_product(const MatrixA<Scalar, MatrixLayoutA>& A, const MatrixB<Scalar, MatrixLayoutB>& B, bool A_transposed = false, bool B_transposed = false)
{
  long A_rows = A_transposed ? A.cols() : A.rows();
  long A_cols = A_transposed ? A.rows() : A.cols();
  long B_rows = B_transposed ? B.cols() : B.rows();
  long B_cols = B_transposed ? B.rows() : B.cols();
  long C_rows = A_rows;
  long C_cols = B_cols;

  assert(A_cols == B_rows);

  constexpr int MatrixLayoutC = (MatrixLayoutA == row_major && MatrixLayoutB == row_major) ? row_major : column_major;
  constexpr CBLAS_LAYOUT cblas_layout = MatrixLayoutC == column_major ? CblasColMajor : CblasRowMajor;
  dense_matrix<Scalar, MatrixLayoutC> C(C_rows, C_cols);
  double alpha = 1.0;
  double beta = 0.0;
  long lda;
  long ldb;
  long ldc;

  if constexpr (MatrixLayoutA == column_major && MatrixLayoutB == column_major)
  {
    lda = A_transposed ? A_cols : A_rows;
    ldb = B_transposed ? B_cols : B_rows;
    ldc = C_rows;
  }
  else if constexpr (MatrixLayoutA == row_major && MatrixLayoutB == row_major)
  {
    lda = A_transposed ? A_rows : A_cols;
    ldb = B_transposed ? B_rows : B_cols;
    ldc = C_cols;
  }
  else if constexpr (MatrixLayoutA == column_major && MatrixLayoutB == row_major)
  {
    auto B_T = make_transposed_dense_matrix_view(B);
    return ddd_product(A, B_T, A_transposed, !B_transposed);
  }
  else if constexpr (MatrixLayoutA == row_major && MatrixLayoutB == column_major)
  {
    auto A_T = make_transposed_dense_matrix_view(A);
    return ddd_product(A_T, B, !A_transposed, B_transposed);
  }

  CBLAS_TRANSPOSE transA = A_transposed ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = B_transposed ? CblasTrans : CblasNoTrans;

  if constexpr (std::is_same<Scalar, double>::value)
  {
    cblas_dgemm(cblas_layout, transA, transB, C_rows, C_cols, A_cols, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
  }
  else
  {
    cblas_sgemm(cblas_layout, transA, transB, C_rows, C_cols, A_cols, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
  }

  return C;
}

template <typename Scalar, int MatrixLayoutA, template <typename, int> class MatrixA, int MatrixLayoutB, template <typename, int> class MatrixB>
dense_matrix<Scalar, column_major> ddd_product_manual_loops(const MatrixA<Scalar, MatrixLayoutA>& A, const MatrixB<Scalar, MatrixLayoutB>& B)
{
  assert(A.cols() == B.rows());

  long m = A.rows();
  long p = A.cols();
  long n = B.cols();

  // returns the dot product of the i-th row of A1 and the j-th column of B1
  auto dot = [&A, &B, p](long i, long j)
  {
    Scalar result = 0;
    for (long k = 0; k < p; k++)
    {
      result += A(i, k) * B(k, j);
    }
    return result;
  };

  dense_matrix<Scalar, column_major> C(m, n);
  for (long i = 0; i < m; i++)
  {
    for (long j = 0; j < n; j++)
    {
      C(i, j) = dot(i, j);
    }
  }

  return C;
}

// Computes the matrix product C = A * B
template <int MatrixLayoutA, typename Scalar, template <typename, int> class MatrixA, int MatrixLayoutB, template <typename, int> class MatrixB>
auto ddd_product_manual_loops(const MatrixA<Scalar, MatrixLayoutA>& A, const MatrixB<Scalar, MatrixLayoutB>& B, bool A_transposed, bool B_transposed)
{
  if (!A_transposed && !B_transposed)
  {
    return ddd_product_manual_loops(A, B);
  }
  else if (!A_transposed && B_transposed)
  {
    return ddd_product_manual_loops(A, make_transposed_dense_matrix_view(B));
  }
  else if (A_transposed && !B_transposed)
  {
    return ddd_product_manual_loops(make_transposed_dense_matrix_view(A), B);
  }
  else
  {
    return ddd_product_manual_loops(make_transposed_dense_matrix_view(A), make_transposed_dense_matrix_view(B));
  }
}

} // namespace nerva::mkl

