// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_dense_matrix.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/mkl_dense_vector.h"
#include "nerva/neural_networks/print_matrix.h"
#include "Eigen/Dense"
#include <mkl.h>
#include <algorithm>
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
std::string matrix_layout_string(int layout)
{
  return layout == column_major ? "C" : "R";
}

inline
std::string matrix_layout_string(int layout1, int layout2)
{
  return matrix_layout_string(layout1) + matrix_layout_string(layout2);
}

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
template <typename Scalar_, int MatrixLayout>
class dense_matrix_view
{
  public:
    using Scalar = Scalar_;

  protected:
    Scalar* m_data;
    long m_rows;
    long m_columns;

  public:
    dense_matrix_view(Scalar* data, long rows, long columns)
      : m_data(data), m_rows(rows), m_columns(columns)
    {
    }

    [[nodiscard]] long rows() const
    {
      return m_rows;
    }

    [[nodiscard]] long cols() const
    {
      return m_columns;
    }

    [[nodiscard]] long leading_dimension() const
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_rows;
      }
      else
      {
        return m_columns;
      }
    }

    Scalar* data()
    {
      return m_data;
    }

    const Scalar* data() const
    {
      return m_data;
    }

    [[nodiscard]] constexpr long offset() const
    {
      return 0;
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

    // N.B. It turns out that `mkl_?imatcopy` is extremely inefficient.
    // See also https://softwarerecs.stackexchange.com/questions/69323/fast-inplace-matrix-transpose-library
    void transpose_in_place()
    {
      constexpr char ordering = (MatrixLayout == column_major ? 'c' : 'r');
      long lda = (MatrixLayout == column_major ? m_rows : m_columns);
      long ldb = (MatrixLayout == column_major ? m_columns : m_rows);
      if constexpr (std::is_same_v<Scalar, double>)
      {
        mkl_dimatcopy(ordering, 't', m_rows, m_columns, 1.0, m_data, lda, ldb);
      }
      else
      {
        mkl_simatcopy(ordering, 't', m_rows, m_columns, 1.0, m_data, lda, ldb);
      }
      std::swap(m_rows, m_columns);
    }

    bool operator==(const dense_matrix_view<Scalar, MatrixLayout>& other) const
    {
      auto size = m_rows * m_columns;
      return m_rows == other.m_rows &&
             m_columns == other.m_columns &&
             std::equal(m_data, m_data + size, other.m_data, other.m_data + size);
    }
};

template <typename Scalar_, int MatrixLayout>
class dense_matrix
{
  public:
    using Scalar = Scalar_;

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

    [[nodiscard]] long leading_dimension() const
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_rows;
      }
      else
      {
        return m_columns;
      }
    }

    Scalar* data()
    {
      return m_data.data();
    }

    const Scalar* data() const
    {
      return m_data.data();
    }

    [[nodiscard]] constexpr long offset() const
    {
      return 0;
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

    // N.B. It turns out that `mkl_?imatcopy` is extremely inefficient.
    // See also https://softwarerecs.stackexchange.com/questions/69323/fast-inplace-matrix-transpose-library
    void transpose_in_place()
    {
      constexpr char ordering = (MatrixLayout == column_major ? 'c' : 'r');
      long lda = (MatrixLayout == column_major ? m_rows : m_columns);
      long ldb = (MatrixLayout == column_major ? m_columns : m_rows);
      if constexpr (std::is_same_v<Scalar, double>)
      {
        mkl_dimatcopy(ordering, 't', m_rows, m_columns, 1.0, m_data, lda, ldb);
      }
      else
      {
        mkl_simatcopy(ordering, 't', m_rows, m_columns, 1.0, m_data, lda, ldb);
      }
      std::swap(m_rows, m_columns);
    }
};

// Represents the submatrix with row indices in the interval [rowmin, ..., rowmax)
template <typename Scalar_, int MatrixLayout>
class dense_submatrix_view
{
  public:
    using Scalar = Scalar_;

  protected:
    dense_matrix_view<Scalar, MatrixLayout> A;
    long m_first_row;
    long m_first_column;
    long m_rows;
    long m_columns;

  public:
    dense_submatrix_view(Scalar* A_data, long A_rows, long A_columns, long first_row, long first_column, long rows, long columns)
      : A(A_data, A_rows, A_columns), m_first_row(first_row), m_first_column(first_column), m_rows(rows), m_columns(columns)
    {
      assert(A(first_row, first_column) == *(data() + offset()));
    }

    [[nodiscard]] long rows() const
    {
      return m_rows;
    }

    [[nodiscard]] long cols() const
    {
      return m_columns;
    }

    [[nodiscard]] long first_row() const
    {
      return m_first_row;
    }

    [[nodiscard]] long first_column() const
    {
      return m_first_column;
    }

    [[nodiscard]] long leading_dimension() const
    {
      if constexpr (MatrixLayout == column_major)
      {
        return A.rows();
      }
      else
      {
        return A.cols();
      }
    }

    Scalar* data()
    {
      return A.data();
    }

    const Scalar* data() const
    {
      return A.data();
    }

    [[nodiscard]] long offset() const
    {
      if constexpr (MatrixLayout == column_major)
      {
        return m_first_row + m_first_column * A.rows();
      }
      else
      {
        return m_first_column + m_first_row * A.cols();
      }
    }

    Scalar operator()(long i, long j) const
    {
      return A(i + m_first_row, j + m_first_column);
    }

    Scalar& operator()(long i, long j)
    {
      return A(i + m_first_row, j + m_first_column);
    }
};

template <typename Scalar, int MatrixLayout>
dense_submatrix_view<Scalar, MatrixLayout> make_dense_matrix_rows_view(dense_matrix_view<Scalar, MatrixLayout>& A, long minrow, long maxrow)
{
  return dense_submatrix_view<Scalar, MatrixLayout>(A.data(), A.rows(), A.cols(), minrow, 0, maxrow - minrow, A.cols());
}

template <typename Scalar, int MatrixLayout>
dense_submatrix_view<Scalar, MatrixLayout> make_dense_matrix_columns_view(dense_matrix_view<Scalar, MatrixLayout>& A, long mincol, long maxcol)
{
  return dense_submatrix_view<Scalar, MatrixLayout>(A.data(), A.rows(), A.cols(), 0, mincol, A.rows(), maxcol - mincol);
}

template <typename Scalar, int MatrixLayout, template <typename, int> class Matrix>
dense_matrix_view<Scalar, MatrixLayout> make_dense_matrix_view(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  return dense_matrix_view<Scalar, MatrixLayout>(const_cast<Scalar*>(A.data()), A.rows(), A.cols());
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

template <typename Scalar, int MatrixLayout>
dense_vector_view<Scalar> make_dense_vector_view(const dense_matrix_view<Scalar, MatrixLayout>& A, long increment = 1)
{
  return dense_vector_view<Scalar>(const_cast<Scalar*>(A.data()), A.rows() * A.cols(), increment);
}

// Computes `B := alpha * op(A)`. This function can for example be used to change the layout of a matrix.
// N.B. A and B should be different!
// The rows and columns of A are not being set.
// The operation op can take the value `n` (unchanged) or `t` (transposed).
// See also https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/mkl-omatcopy.html
template <typename Scalar, int MatrixLayout, template <typename, int> class MatrixA, template <typename, int> class MatrixB>
void matrix_copy(const MatrixA<Scalar, MatrixLayout>& A, MatrixB<Scalar, MatrixLayout>& B, Scalar alpha, char op)
{
  constexpr char ordering = (MatrixLayout == column_major ? 'c' : 'r');
  long lda = A.leading_dimension(); // (MatrixLayout == column_major ? m_rows : m_columns);
  long ldb = B.leading_dimension(); // (MatrixLayout == column_major ? m_columns : m_rows);
  if constexpr (std::is_same_v<Scalar, double>)
  {
    mkl_domatcopy(ordering, op, A.rows(), A.cols(), alpha, A.data(), lda, B.data(), ldb);
  }
  else
  {
    mkl_somatcopy(ordering, op, A.rows(), A.cols(), alpha, A.data(), lda, B.data(), ldb);
  }
}

// Computes B := A, such that B is a copy of A, but with different layout.
template <typename Scalar, int MatrixLayout, template <typename, int> class MatrixA, template <typename, int> class MatrixB>
void change_matrix_layout(const MatrixA<Scalar, MatrixLayout>& A, MatrixB<Scalar, 1 - MatrixLayout>& B)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == B.cols());
  dense_matrix_view<Scalar, MatrixLayout> B1(B.data(), B.cols(), B.rows());
  matrix_copy(A, B1, Scalar(1), 't');
}

// Computes the matrix product C := A * B
template <typename Scalar, int MatrixLayoutC, template <typename, int> class MatrixC, int MatrixLayoutA, template <typename, int> class MatrixA, int MatrixLayoutB, template <typename, int> class MatrixB>
void ddd_product(MatrixC<Scalar, MatrixLayoutC>& C, const MatrixA<Scalar, MatrixLayoutA>& A, const MatrixB<Scalar, MatrixLayoutB>& B, bool A_transposed = false, bool B_transposed = false)
{
  if constexpr (MatrixLayoutA != MatrixLayoutC && MatrixLayoutB != MatrixLayoutC)
  {
    auto A_T = make_transposed_dense_matrix_view(A);
    auto B_T = make_transposed_dense_matrix_view(B);
    ddd_product(C, A_T, B_T, !A_transposed, !B_transposed);
    return;
  }
  else if constexpr (MatrixLayoutA != MatrixLayoutC)
  {
    auto A_T = make_transposed_dense_matrix_view(A);
    ddd_product(C, A_T, B, !A_transposed, B_transposed);
    return;
  }
  else if constexpr (MatrixLayoutB != MatrixLayoutC)
  {
    auto B_T = make_transposed_dense_matrix_view(B);
    ddd_product(C, A, B_T, A_transposed, !B_transposed);
    return;
  }

  [[maybe_unused]] long A_rows = A_transposed ? A.cols() : A.rows();
  [[maybe_unused]] long A_cols = A_transposed ? A.rows() : A.cols();
  [[maybe_unused]] long B_rows = B_transposed ? B.cols() : B.rows();
  [[maybe_unused]] long B_cols = B_transposed ? B.rows() : B.cols();
  [[maybe_unused]] long C_rows = A_rows;
  [[maybe_unused]] long C_cols = B_cols;
  assert(A_cols == B_rows);

  long m = A_transposed ? A.cols() : A.rows();
  long n = B_transposed ? B.rows() : B.cols();
  long k = A_transposed ? A.rows() : A.cols();

  assert(C.rows() >= m);
  assert(C.cols() >= n);

  constexpr CBLAS_LAYOUT cblas_layout = MatrixLayoutC == column_major ? CblasColMajor : CblasRowMajor;
  double alpha = 1.0;
  double beta = 0.0;
  long lda = A.leading_dimension();
  long ldb = B.leading_dimension();
  long ldc = C.leading_dimension();

  CBLAS_TRANSPOSE transA = A_transposed ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = B_transposed ? CblasTrans : CblasNoTrans;

  if constexpr (std::is_same<Scalar, double>::value)
  {
    cblas_dgemm(cblas_layout, transA, transB, m, n, k, alpha, A.data() + A.offset(), lda, B.data() + B.offset(), ldb, beta, C.data() + C.offset(), ldc);
  }
  else
  {
    cblas_sgemm(cblas_layout, transA, transB, m, n, k, alpha, A.data() + A.offset(), lda, B.data() + B.offset(), ldb, beta, C.data() + C.offset(), ldc);
  }
}

// Computes the matrix product C = A * B
template <typename Scalar, int MatrixLayoutA, template <typename, int> class MatrixA, int MatrixLayoutB, template <typename, int> class MatrixB>
auto ddd_product_copy(const MatrixA<Scalar, MatrixLayoutA>& A, const MatrixB<Scalar, MatrixLayoutB>& B, bool A_transposed = false, bool B_transposed = false)
{
  constexpr int MatrixLayoutC = MatrixLayoutA;
  long C_rows = A_transposed ? A.cols() : A.rows();
  long C_cols = B_transposed ? B.rows() : B.cols();
  dense_matrix<Scalar, MatrixLayoutC> C(C_rows, C_cols);
  ddd_product(C, A, B, A_transposed, B_transposed);
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

//----------------------------------------------------------------------//
//                     BLAS level 1 routines
//----------------------------------------------------------------------//

// y := a*x + y
template <typename Scalar, int MatrixLayout>
void cblas_axpy(Scalar a, const dense_matrix_view<Scalar, MatrixLayout>& x, dense_matrix_view<Scalar, MatrixLayout>& y)
{
  assert(x.rows() == y.rows() && x.cols() == y.cols());
  auto x_view = make_dense_vector_view(x);
  auto y_view = make_dense_vector_view(y);
  return cblas_axpy(a, x_view, y_view);
};

// returns elements_sum(abs(x))
template <typename Scalar, int MatrixLayout>
Scalar cblas_asum(dense_matrix_view<Scalar, MatrixLayout>& x)
{
  auto x_view = make_dense_vector_view(x);
  return cblas_asum(x_view);
};

// y := x
template <typename Scalar, int MatrixLayout>
void cblas_copy(const dense_matrix_view<Scalar, MatrixLayout>& x, dense_matrix_view<Scalar, MatrixLayout>& y)
{
  assert(x.rows() == y.rows() && x.cols() == y.cols());
  auto x_view = make_dense_vector_view(x);
  auto y_view = make_dense_vector_view(y);
  cblas_copy(x_view, y_view);
};

// returns elements_sum(hadamard(x, y))
template <typename Scalar, int MatrixLayout>
Scalar cblas_dot(const dense_matrix_view<Scalar, MatrixLayout>& x, dense_matrix_view<Scalar, MatrixLayout>& y)
{
  auto x_view = make_dense_vector_view(x);
  auto y_view = make_dense_vector_view(y);
  return cblas_dot(x_view, y_view);
};

// returns elements_sum(abs(sqrt(x)))
template <typename Scalar, int MatrixLayout>
Scalar cblas_nrm2(dense_matrix_view<Scalar, MatrixLayout>& x)
{
  auto x_view = make_dense_vector_view(x);
  cblas_nrm2(x_view);
};

// x := a*x
template <typename Scalar, int MatrixLayout>
void cblas_scal(Scalar a, dense_matrix_view<Scalar, MatrixLayout>& x)
{
  auto x_view = make_dense_vector_view(x);
  cblas_scal(a, x_view);
};

// x, y := y, x
template <typename Scalar, int MatrixLayout>
void cblas_swap(const dense_matrix_view<Scalar, MatrixLayout>& x, dense_matrix_view<Scalar, MatrixLayout>& y)
{
  auto x_view = make_dense_vector_view(x);
  auto y_view = make_dense_vector_view(y);
  cblas_swap(x_view, y_view);
};

// Returns the index of the element with maximum absolute value.
template <typename Scalar, int MatrixLayout>
CBLAS_INDEX cblas_iamax(dense_matrix_view<Scalar, MatrixLayout>& x)
{
  auto x_view = make_dense_vector_view(x);
  return cblas_iamax(x_view);
};

// Returns the index of the element with minimum absolute value.
template <typename Scalar, int MatrixLayout>
CBLAS_INDEX cblas_iamin(dense_matrix_view<Scalar, MatrixLayout>& x)
{
  auto x_view = make_dense_vector_view(x);
  return cblas_iamin(x_view);
};

//----------------------------------------------------------------------//
//                     BLAS level 2 routines
//----------------------------------------------------------------------//

// y := alpha*A*x + beta*y
template <typename Scalar, int MatrixLayout>
void cblas_gemv(Scalar alpha,
                Scalar beta,
                const dense_matrix_view<Scalar, MatrixLayout>& A,
                const dense_vector_view<Scalar>& x,
                dense_vector_view<Scalar>& y)
{
  constexpr CBLAS_LAYOUT layoutA = MatrixLayout == column_major ? CblasColMajor : CblasRowMajor;
  constexpr CBLAS_TRANSPOSE transA = CblasNoTrans;

  if constexpr (std::is_same_v<Scalar, float>)
  {
    cblas_sgemv(layoutA, transA, A.rows(), A.cols(), alpha, A.data(), A.leading_dimension(), x.data(), x.increment(), beta, y.data(), y.increment());
  }
  else
  {
    cblas_dgemv(layoutA, transA, A.rows(), A.cols(), alpha, A.data(), A.leading_dimension(), x.data(), x.increment(), beta, y.data(), y.increment());
  }
}

// A := alpha * x * y^T + A
template <typename Scalar, int MatrixLayout>
void cblas_ger(Scalar alpha,
               const dense_vector_view<Scalar>& x,
               const dense_vector_view<Scalar>& y,
               dense_matrix_view<Scalar, MatrixLayout>& A)
{
  constexpr CBLAS_LAYOUT layoutA = MatrixLayout == column_major ? CblasColMajor : CblasRowMajor;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    cblas_sger(layoutA, A.rows(), A.cols(), alpha, A.data(), x.data(), x.increment(), y.data(), y.increment(), A.data(), A.leading_dimension());
  }
  else
  {
    cblas_dger(layoutA, A.rows(), A.cols(), alpha, A.data(), x.data(), x.increment(), y.data(), y.increment(), A.data(), A.leading_dimension());
  }
}

//----------------------------------------------------------------------//
//                     BLAS level 3 routines
//----------------------------------------------------------------------//

// C := alpha*op(A)*op(B) + beta*C
template <typename Scalar, int MatrixLayout>
void cblas_gemm(const dense_matrix_view<Scalar, MatrixLayout>& A,
                bool A_transposed,
                const dense_matrix_view<Scalar, MatrixLayout>& B,
                bool B_transposed,
                dense_matrix_view<Scalar, MatrixLayout>& C,
                Scalar alpha,
                Scalar beta
               )
{
  constexpr CBLAS_LAYOUT layout = MatrixLayout == column_major ? CblasColMajor : CblasRowMajor;
  CBLAS_TRANSPOSE transA = A_transposed ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = B_transposed ? CblasTrans : CblasNoTrans;

  // op(A) is an m-by-k matrix,
  // op(B) is a k-by-n matrix,
  // C is an m-by-n matrix.
  long m = A_transposed ? A.cols() : A.rows();
  long k = A_transposed ? A.rows() : A.cols();
  long n = B_transposed ? B.rows() : B.cols();

  if constexpr (std::is_same_v<Scalar, float>)
  {
    cblas_sgemm(layout, transA, transB, m, n, k, alpha, A.data(), A.leading_dimension(), B.data(), B.leading_dimension(), beta, C.data(), C.leading_dimension());
  }
  else
  {
    cblas_dgemm(layout, transA, transB, m, n, k, alpha, A.data(), A.leading_dimension(), B.data(), B.leading_dimension(), beta, C.data(), C.leading_dimension());
  }
}


} // namespace nerva::mkl

