// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_sparse_matrix.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/functions.h"
#include "nerva/neural_networks/mkl_dense_matrix.h"
#include "nerva/utilities/print.h"
#include "nerva/utilities/random.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerva::mkl {

// prototype declarations
template <typename Scalar> class sparse_matrix_csr;
template <typename Scalar> void compare_sizes(const mkl::sparse_matrix_csr<Scalar>& A, const mkl::sparse_matrix_csr<Scalar>& B);

inline
std::string sparse_status_message(sparse_status_t status)
{
  switch(status)
  {
    case SPARSE_STATUS_SUCCESS: return "The operation was successful.";
    case SPARSE_STATUS_NOT_INITIALIZED: return "The routine encountered an empty handle or matrix array.";
    case SPARSE_STATUS_ALLOC_FAILED: return "Internal memory allocation failed.";
    case SPARSE_STATUS_INVALID_VALUE: return "The input parameters contain an invalid value.";
    case SPARSE_STATUS_EXECUTION_FAILED: return "Execution failed.";
    case SPARSE_STATUS_INTERNAL_ERROR: return "An error in algorithm implementation occurred.";
    case SPARSE_STATUS_NOT_SUPPORTED: return "The requested operation is not supported.";
  }
  throw std::runtime_error("unknown sparse_status_t value " + std::to_string(status));
}

// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-csr-matrix-storage-format.html
template <typename T>
class sparse_matrix_csr
{
  public:
    using Scalar = T;

  protected:
    long m_rows{}; // number of rows
    long m_columns{}; // number of columns
    std::vector<MKL_INT> m_row_index;
    std::vector<MKL_INT> m_col_index;
    std::vector<T> m_values;
    // The sparse_matrix_t type used by the Intel MKL library is an opaque type,
    // which means that its internal structure is not exposed to the user. This
    // type is used to represent sparse matrices in the MKL library and is used as
    // a handle for various sparse matrix operations. Whenever the content of the
    // attributes row_index, columns or values is changed, the csr object needs to
    // be recreated(!!!).
    sparse_matrix_t m_csr{nullptr};
    matrix_descr m_descr{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};

    [[nodiscard]] bool is_valid() const
    {
      // Check if dimensions are non-negative
      if (m_rows < 0 || m_columns < 0)
      {
        return false;
      }

      // Check if row index size matches the number of rows + 1
      if (m_row_index.size() != static_cast<std::size_t>(m_rows) + 1)
      {
        return false;
      }

      // Check if columns and values have the same size
      if (m_col_index.size() != m_values.size())
      {
        return false;
      }

      // Check if row index values are non-negative and non-decreasing
      for (size_t i = 0; i < m_row_index.size() - 1; i++)
      {
        if (m_row_index[i] < 0 || m_row_index[i] > static_cast<long>(m_values.size()))
        {
          return false;
        }
        if (m_row_index[i] > m_row_index[i + 1])
        {
          return false;
        }
      }

      // Check if column indices are within bounds
      for (auto column: m_col_index)
      {
        if (column < 0 || column >= m_columns)
        {
          return false;
        }
      }

      // Check if the data() pointers of columns and values are defined
      if (m_col_index.capacity() == 0 || m_values.capacity() == 0)
      {
        return false;
      }

      // If all checks passed, the sparse matrix is valid
      return true;
    }

    void destruct_csr()
    {
      if (m_csr)
      {
        mkl_sparse_destroy(m_csr);
      }
    }

  public:
    void construct_csr(bool throw_on_error = true)
    {
      assert(is_valid());
      destruct_csr();
      sparse_status_t status;
      if constexpr (std::is_same<T, double>::value)
      {
        status = mkl_sparse_d_create_csr(&m_csr,
                                         SPARSE_INDEX_BASE_ZERO,
                                         m_rows,
                                         m_columns,
                                         m_row_index.data(),
                                         m_row_index.data() + 1,
                                         m_col_index.data(),
                                         m_values.data());
      }
      else
      {
        status = mkl_sparse_s_create_csr(&m_csr,
                                         SPARSE_INDEX_BASE_ZERO,
                                         m_rows,
                                         m_columns,
                                         m_row_index.data(),
                                         m_row_index.data() + 1,
                                         m_col_index.data(),
                                         m_values.data());
      }
      if (status != SPARSE_STATUS_SUCCESS)
      {
        std::string error_message = "mkl_sparse_?_create_csr: " + sparse_status_message(status);
        if (throw_on_error)
        {
          throw std::runtime_error(error_message);
        }
        else
        {
          std::cout << "Error: " + error_message << std::endl;
          std::exit(1);
        }
      }
    }

    // Creates a sparse matrix with empty support
    explicit sparse_matrix_csr(long rows = 1, long cols = 1, std::size_t size = 1)
      : m_rows(rows), m_columns(cols), m_row_index(rows + 1, 0)
    {
      assert(size > 0);  // this is needed due to a limitation of MKL
      m_col_index.reserve(size);
      m_values.reserve(size);
      construct_csr();
    }

    sparse_matrix_csr(long rows,
                      long cols,
                      std::vector<MKL_INT> row_index_,
                      std::vector<MKL_INT> columns_,
                      std::vector<T> values_
    )
      : m_rows(rows), m_columns(cols), m_row_index(std::move(row_index_)), m_col_index(std::move(columns_)), m_values(std::move(values_))
    {
      construct_csr();
    }

    sparse_matrix_csr(const sparse_matrix_csr& A)
     : m_rows(A.m_rows),
       m_columns(A.m_columns),
       m_row_index(A.m_row_index),
       m_col_index(A.m_col_index),
       m_values(A.m_values)
    {
      construct_csr(false);
    }

    sparse_matrix_csr(sparse_matrix_csr&& A) noexcept
     : m_rows(A.m_rows),
       m_columns(A.m_columns),
       m_row_index(std::move(A.m_row_index)),
       m_col_index(std::move(A.m_col_index)),
       m_values(std::move(A.m_values))
    {
      construct_csr(false);
    }

    sparse_matrix_csr& operator=(const sparse_matrix_csr& A)
    {
      // N.B. No move operations are used, since we want to keep using the originally allocated memory
      m_rows = A.m_rows;
      m_columns = A.m_columns;
      m_row_index = A.m_row_index;
      m_col_index = A.m_col_index;
      m_values = A.m_values;
      m_descr = A.m_descr;
      construct_csr();
      return *this;
    }

    ~sparse_matrix_csr()
    {
      destruct_csr();
    }

    [[nodiscard]] long rows() const
    {
      return m_rows;
    }

    [[nodiscard]] long cols() const
    {
      return m_columns;
    }

    [[nodiscard]] const std::vector<MKL_INT>& row_index() const
    {
      return m_row_index;
    }
    
    [[nodiscard]] const std::vector<MKL_INT>& col_index() const
    {
      return m_col_index;
    }
    
    const std::vector<T>& values() const
    {
      return m_values;
    }

    std::vector<T>& values()
    {
      return m_values;
    }

    [[nodiscard]] const matrix_descr& descriptor() const
    {
      return m_descr;
    }

    [[nodiscard]] sparse_matrix_t csr() const
    {
      return m_csr;
    }

    [[nodiscard]] Scalar density() const
    {
      return Scalar(m_values.size()) / (m_rows * m_columns);
    }

    // Copies the support set of other and sets all values to 0.
    // The support size must match.
    void reset_support(const sparse_matrix_csr& other)
    {
      compare_sizes(*this, other);
      if (m_values.size() == other.m_values.size())
      {
        std::fill(m_values.begin(), m_values.end(), 0);
        std::copy(other.m_col_index.begin(), other.m_col_index.end(), m_col_index.begin());
        std::copy(other.m_row_index.begin(), other.m_row_index.end(), m_row_index.begin());
      }
      else
      {
        m_rows = other.m_rows;
        m_columns = other.m_columns;
        m_row_index = other.m_row_index;
        m_col_index = other.m_col_index;
        m_values = std::vector<T>(other.m_values.size(), 0);
      }
      construct_csr();
    }

    // Assign the given value to all coefficients
    sparse_matrix_csr& operator=(T value)
    {
      std::fill(m_values.begin(), m_values.end(), value);
      construct_csr();
      return *this;
    }

    [[nodiscard]] std::string to_string() const
    {
      std::ostringstream out;
      out << "--- mkl matrix ---\n";
      out << "dimension: " << m_rows << " x " << m_columns << '\n';
      out << "values:    " << m_values.size() << ' ' << nerva::print_list(m_values) << '\n';
      out << "columns:   " << m_col_index.size() << ' ' << nerva::print_list(m_col_index) << '\n';
      out << "row_index: " << m_row_index.size() << ' ' << nerva::print_list(m_row_index) << '\n';
      return out.str();
    }

    template <typename Scalar> friend void assign_matrix(sparse_matrix_csr<Scalar>& A, const sparse_matrix_csr<Scalar>& B);
    template <typename Scalar, typename Function> friend void initialize_matrix(sparse_matrix_csr<Scalar>& A, Function f);
};

template <typename T>
std::size_t support_size(const sparse_matrix_csr<T>& A)
{
  return A.values().size();
}

template <typename T>
std::size_t count_positive_elements(const sparse_matrix_csr<T>& A)
{
  const auto& values = A.values();
  return std::count_if(values.begin(), values.end(), [](auto x) { return x > 0; });
}

template <typename T>
std::size_t count_negative_elements(const sparse_matrix_csr<T>& A)
{
  const auto& values = A.values();
  return std::count_if(values.begin(), values.end(), [](auto x) { return x < 0; });
}

// Does the assignment A := alpha * A + beta * op(B) * C with B sparse and A, C dense
//
// operation_B determines whether op(B) = B or op(B) = B^T
template <typename Scalar, int MatrixLayout>
void dsd_product(dense_matrix_view<Scalar, MatrixLayout>& A,
                 const mkl::sparse_matrix_csr<Scalar>& B,
                 const dense_matrix_view<Scalar, MatrixLayout>& C,
                 Scalar alpha = 0,
                 Scalar beta = 1,
                 sparse_operation_t operation_B = SPARSE_OPERATION_NON_TRANSPOSE
)
{
  assert(A.rows() == (operation_B == SPARSE_OPERATION_NON_TRANSPOSE ? B.rows() : B.cols()));
  assert(A.cols() == C.cols());
  assert((operation_B == SPARSE_OPERATION_NON_TRANSPOSE ? B.cols() : B.rows()) == C.rows());

  sparse_status_t status;
  if constexpr (std::is_same<Scalar, double>::value)
  {
    if constexpr (MatrixLayout == matrix_layout::column_major)
    {
      status = mkl_sparse_d_mm(operation_B, beta, B.csr(), B.descriptor(), SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), A.cols(), C.rows(), alpha, A.data(), A.rows());
    }
    else
    {
      status = mkl_sparse_d_mm(operation_B, beta, B.csr(), B.descriptor(), SPARSE_LAYOUT_ROW_MAJOR, C.data(), A.cols(), C.cols(), alpha, A.data(), A.cols());
    }
  }
  else
  {
    if constexpr (MatrixLayout == matrix_layout::column_major)
    {
      status = mkl_sparse_s_mm(operation_B, beta, B.csr(), B.descriptor(), SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), A.cols(), C.rows(), alpha, A.data(), A.rows());
    }
    else
    {
      status = mkl_sparse_s_mm(operation_B, beta, B.csr(), B.descriptor(), SPARSE_LAYOUT_ROW_MAJOR, C.data(), A.cols(), C.cols(), alpha, A.data(), A.cols());
    }
  }

  if (status != SPARSE_STATUS_SUCCESS)
  {
    std::string error_message = "mkl_sparse_?_mm: " + sparse_status_message(status);
    throw std::runtime_error(error_message);
  }
}

// Performs the assignment A := B, with A, B sparse. A and B must have the same support.
template <typename Scalar>
void assign_matrix(sparse_matrix_csr<Scalar>& A, const sparse_matrix_csr<Scalar>& B)
{
  std::copy(B.m_values.begin(), B.m_values.end(), A.m_values.begin());
  A.construct_csr();
}

template <typename Scalar, typename Function>
void initialize_matrix(sparse_matrix_csr<Scalar>& A, Function f)
{
  for (auto& value: A.m_values)
  {
    value = f();
  }
  A.construct_csr();
}

template <typename Scalar>
void compare_sizes(const mkl::sparse_matrix_csr<Scalar>& A, const mkl::sparse_matrix_csr<Scalar>& B)
{
  if (A.rows() != B.rows() || A.cols() != B.cols())
  {
    throw std::runtime_error("matrix sizes do not match");
  }
}

template <typename Scalar>
struct csr_matrix_builder
{
  long rows;
  long columns;
  std::vector<MKL_INT> row_index;
  std::vector<MKL_INT> col_index;
  std::vector<Scalar> values;

  csr_matrix_builder(long rows_, long columns_, std::size_t size = 1)
   : rows(rows_), columns(columns_)
  {
    row_index.reserve(rows + 1);
    col_index.reserve(size);
    values.reserve(size);
  }

  void add_element(long i, long j, Scalar value)
  {
    while (static_cast<std::size_t>(i) >= row_index.size())
    {
      row_index.push_back(values.size());
    }
    col_index.push_back(j);
    values.push_back(value);
  }

  mkl::sparse_matrix_csr<Scalar> result()
  {
    while (row_index.size() <= static_cast<std::size_t>(rows))
    {
      row_index.push_back(values.size());
    }
    return mkl::sparse_matrix_csr<Scalar>(rows, columns, row_index, col_index, values);
  }
};

/// Creates a random sparse matrix with `nonzero_count` elements. The elements are initialized using the function `f`.
/// \param rows The number of rows of the matrix
/// \param columns The number of columns of the matrix
/// \param nonzero_count The size of the support
/// \param rng A random number generator
/// \param f A function that is used to assign initial values to the elements. By default elements are initialized with the value 0.
template <typename Scalar, typename Function = zero<Scalar>>
mkl::sparse_matrix_csr<Scalar> make_random_matrix(std::size_t rows, std::size_t columns, std::size_t nonzero_count, std::mt19937& rng, Function f = Function())
{
  assert(nonzero_count <= rows * columns);
  mkl::csr_matrix_builder<Scalar> builder(rows, columns, nonzero_count);

  std::size_t remaining_positions = rows * columns;  // the remaining number of positions
  std::size_t remaining_nonzero_positions = nonzero_count;  // the remaining number of nonzero positions

  for (std::size_t i = 0; i < rows; i++)
  {
    for (std::size_t j = 0; j < columns; j++)
    {
      if ((random_real<double>(0, 1, rng) < static_cast<double>(remaining_nonzero_positions) / remaining_positions) || (remaining_positions == remaining_nonzero_positions))
      {
        remaining_nonzero_positions--;
        builder.add_element(i, j, f());
      }
      remaining_positions--;
    }
  }

  return builder.result();
}

// calls f(i, j, A(i,j)) for each valid index (i, j) in A
template <typename T, typename Function>
void traverse_elements(const sparse_matrix_csr<T>& A, Function f)
{
  const auto& row_index = A.row_index();
  const auto& col_index = A.col_index();
  const T* data = A.values().data();
  long m = A.rows();

  for (long i = 0; i < m; i++)
  {
    for (auto k = row_index[i]; k < row_index[i + 1]; k++)
    {
      long j = col_index[k];
      f(i, j, *data++);
    }
  }
}

template <typename T>
bool has_nan(const sparse_matrix_csr<T>& A)
{
  for (auto x: A.values())
  {
    if (std::isnan(x))
    {
      return true;
    }
  }
  return false;
}

} // namespace nerva::mkl

