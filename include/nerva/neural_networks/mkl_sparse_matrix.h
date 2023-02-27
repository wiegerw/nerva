// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_sparse_matrix.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MKL_SPARSE_MATRIX_H
#define NERVA_NEURAL_NETWORKS_MKL_SPARSE_MATRIX_H

#include "nerva/neural_networks/mkl_dense_matrix.h"
#include "nerva/utilities/print.h"
#include "nerva/utilities/random.h"
#include <mkl.h>
#include <mkl_spblas.h>
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
template <typename Scalar> struct sparse_matrix_csr;
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
struct sparse_matrix_csr
{
  using Scalar = T;

  std::vector<MKL_INT> row_index;
  std::vector<MKL_INT> columns;
  std::vector<T> values;

  // The sparse_matrix_t type used by the Intel MKL library is an opaque type,
  // which means that its internal structure is not exposed to the user. This
  // type is used to represent sparse matrices in the MKL library and is used as
  // a handle for various sparse matrix operations. Whenever the content of the
  // attributes row_index, columns or values is changed, the csr object needs to
  // be recreated(!!!).
  sparse_matrix_t csr{nullptr};
  matrix_descr descr{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};

  long m{}; // number of rows
  long n{}; // number of columns

  void check() const
  {
    assert(m > 0);
    assert(n > 0);
    assert(row_index.size() == m + 1);
    assert(columns.size() == values.size());
    assert(columns.capacity() != 0);
  }

  void destruct_csr()
  {
    if (csr)
    {
      mkl_sparse_destroy(csr);
    }
  }

  void construct_csr(bool throw_on_error = true)
  {
    destruct_csr();
    sparse_status_t status;
    if constexpr (std::is_same<T, double>::value)
    {
      status = mkl_sparse_d_create_csr(&csr,
                                       SPARSE_INDEX_BASE_ZERO,
                                       m,
                                       n,
                                       row_index.data(),
                                       row_index.data() + 1,
                                       columns.data(),
                                       values.data());
    }
    else
    {
      status = mkl_sparse_s_create_csr(&csr,
                                       SPARSE_INDEX_BASE_ZERO,
                                       m,
                                       n,
                                       row_index.data(),
                                       row_index.data() + 1,
                                       columns.data(),
                                       values.data());
    }
    if (status != SPARSE_STATUS_SUCCESS)
    {
      std::cout << "rows = " << m << std::endl;
      std::cout << "columns = " << n << std::endl;
      std::cout << "|row_index| = " << row_index.size() << std::endl;
      std::cout << "|columns| = " << columns.size() << std::endl;
      std::cout << "|values| = " << values.size() << std::endl;
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
    check();
  }

  // Creates a sparse matrix with empty support
  explicit sparse_matrix_csr(long rows = 1, long cols = 1)
    : row_index(rows + 1, 0), m(rows), n(cols)
  {
    columns.reserve(1);  // to make sure that the data pointer has a value
    values.reserve(1);   // to make sure that the data pointer has a value
    construct_csr();
  }

  sparse_matrix_csr(long rows,
                    long cols,
                    std::vector<MKL_INT> row_index_,
                    std::vector<MKL_INT> columns_,
                    std::vector<T> values_
  )
    : row_index(std::move(row_index_)), columns(std::move(columns_)), values(std::move(values_)), m(rows), n(cols)
  {
    construct_csr();
  }

  sparse_matrix_csr(long rows, long cols, double density, std::mt19937& rng, T value = 0)
    : m(rows), n(cols)
  {
    long nonzero_count = std::lround(density * rows * cols);
    assert(0 < nonzero_count && nonzero_count <= rows * cols);
    values.reserve(nonzero_count);
    columns.reserve(nonzero_count);
    row_index.push_back(0);

    long remaining = rows * cols;  // the remaining number of positions
    long nonzero = nonzero_count;  // the remaining number of nonzero positions

    for (auto i = 0; i < m; i++)
    {
      long row_count = 0;
      for (auto j = 0; j < n; j++)
      {
        if ((random_real<double>(0, 1, rng) < static_cast<double>(nonzero) / remaining) || (remaining == nonzero))
        {
          columns.push_back(static_cast<long>(j));
          values.push_back(value);
          row_count++;
          nonzero--;
        }
        remaining--;
      }
      row_index.push_back(row_index.back() + row_count);
    }
    construct_csr(false);
  }

  sparse_matrix_csr(const sparse_matrix_csr& A)
    : row_index(A.row_index),
      columns(A.columns),
      values(A.values),
      m(A.m),
      n(A.n)
  {
    construct_csr(false);
  }

  sparse_matrix_csr(sparse_matrix_csr&& A) noexcept
    : row_index(std::move(A.row_index)),
      columns(std::move(A.columns)),
      values(std::move(A.values)),
      m(A.m),
      n(A.n)
  {
    construct_csr(false);
  }

  sparse_matrix_csr& operator=(const sparse_matrix_csr& A)
  {
    row_index = A.row_index;
    columns = A.columns;
    values = A.values;
    m = A.m;
    n = A.n;
    construct_csr();
    return *this;
  }

  ~sparse_matrix_csr()
  {
    destruct_csr();
  }

  // Copies the support set of other and sets all values to 0.
  // The support size must match.
  void reset_support(const sparse_matrix_csr& other)
  {
    compare_sizes(*this, other);
    if (values.size() == other.values.size())
    {
      std::fill(values.begin(), values.end(), 0);
      std::copy(other.columns.begin(), other.columns.end(), columns.begin());
      std::copy(other.row_index.begin(), other.row_index.end(), row_index.begin());
    }
    else
    {
      row_index = other.row_index;
      columns = other.columns;
      values = std::vector<T>(other.values.size(), 0);
      m = other.m;
      n = other.n;
    }
    construct_csr();
  }

  [[nodiscard]] long rows() const
  {
    return m;
  }

  [[nodiscard]] long cols() const
  {
    return n;
  }

  // Assign the given value to all coefficients
  sparse_matrix_csr& operator=(T value)
  {
    std::fill(values.begin(), values.end(), value);
    return *this;
  }

  [[nodiscard]] std::string to_string() const
  {
    std::ostringstream out;
    out << "--- mkl matrix ---\n";
    out << "dimension: " << m << " x " << n << '\n';
    out << "values:    " << values.size() << ' ' << nerva::print_list(values) << '\n';
    out << "columns:   " << columns.size() << ' ' << nerva::print_list(columns) << '\n';
    out << "row_index: " << row_index.size() << ' ' << nerva::print_list(row_index) << '\n';
    return out.str();
  }

  [[nodiscard]] Scalar density() const
  {
    return Scalar(values.size()) / (m * n);
  }
};

// Does the assignment A := alpha * A + beta * op(B) * C with B sparse and A, C dense
//
// operation_B determines whether op(B) = B or op(B) = B^T
template <typename Scalar>
void assign_matrix_product(dense_matrix_view<Scalar>& A,
                           const mkl::sparse_matrix_csr<Scalar>& B,
                           const dense_matrix_view<Scalar>& C,
                           Scalar alpha = 0,
                           Scalar beta = 1,
                           sparse_operation_t operation_B = SPARSE_OPERATION_NON_TRANSPOSE
)
{
  assert(A.rows() == (operation_B == SPARSE_OPERATION_NON_TRANSPOSE ? B.rows() : B.cols()));
  assert(A.cols() == C.cols());
  assert((operation_B == SPARSE_OPERATION_NON_TRANSPOSE ? B.cols() : B.rows()) == C.rows());
  assert(A.layout() == C.layout());

  sparse_status_t status;
  if constexpr (std::is_same<Scalar, double>::value)
  {
    if (A.layout() == matrix_layout::column_major)
    {
      status = mkl_sparse_d_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), A.cols(), C.rows(), alpha, A.data(), A.rows());
    }
    else
    {
      status = mkl_sparse_d_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_ROW_MAJOR, C.data(), A.cols(), C.cols(), alpha, A.data(), A.cols());
    }
  }
  else
  {
    if (A.layout() == matrix_layout::column_major)
    {
      status = mkl_sparse_s_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), A.cols(), C.rows(), alpha, A.data(), A.rows());
    }
    else
    {
      status = mkl_sparse_s_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_ROW_MAJOR, C.data(), A.cols(), C.cols(), alpha, A.data(), A.cols());
    }
  }

  if (status != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("mkl_sparse_dense_mat_mult reported status " + std::to_string(status));
  }
}

// Performs the assignment A := B, with A, B sparse. A and B must have the same support.
template <typename Scalar>
void assign_matrix(sparse_matrix_csr<Scalar>& A, const sparse_matrix_csr<Scalar>& B)
{
  std::copy(B.values.begin(), B.values.end(), A.values.begin());
  A.construct_csr();
}

template <typename Scalar, typename Function>
void initialize_matrix(sparse_matrix_csr<Scalar>& A, Function f)
{
  for (auto& value: A.values)
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

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_SPARSE_MATRIX_H
