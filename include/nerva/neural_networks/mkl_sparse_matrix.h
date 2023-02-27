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
    if (m > 0 && n > 0)
    {
      assert(row_index.size() == m + 1);
      assert(columns.size() == values.size());
    }
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

  explicit sparse_matrix_csr(long rows = 0, long columns = 0)
    : m(rows), n(columns)
  {
    // N.B. This brings the matrix in an unusable state.
    // TODO: find out if MKL allows a useful default initialization
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

  sparse_matrix_csr& assign(const sparse_matrix_csr& A, T value = 0)
  {
    row_index = A.row_index;
    columns = A.columns;
    values = std::vector<T>(A.values.size(), value);
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
      std::cout << "reset_support |values| = " + std::to_string(values.size()) + " |values'| = " + std::to_string(other.values.size()) << std::endl;
      assign(other, 0);
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

// Does the assignment A := alpha * A + beta * B, with A, B sparse.
// A and B must have the same non-zero mask
template <typename Scalar = scalar>
void assign_matrix_sum(mkl::sparse_matrix_csr<Scalar>& A,
                       const mkl::sparse_matrix_csr<Scalar>& B,
                       Scalar alpha = 0.0,
                       Scalar beta = 1.0
)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == B.cols());
  assert(A.columns == B.columns);
  assert(A.row_index == B.row_index);

  eigen::vector_map<Scalar> A1(const_cast<Scalar*>(A.values.data()), A.values.size());
  eigen::vector_map<Scalar> B1(const_cast<Scalar*>(B.values.data()), B.values.size());

  A1 = alpha * A1 + beta * B1;
  A.construct_csr();
}

// Does the assignment A := alpha * A + beta * B + gamma * C, with A, B, C sparse.
// A, B and C must have the same support
template <typename Scalar = scalar>
void assign_matrix_sum(mkl::sparse_matrix_csr<Scalar>& A,
                       const mkl::sparse_matrix_csr<Scalar>& B,
                       const mkl::sparse_matrix_csr<Scalar>& C,
                       Scalar alpha = 1.0,
                       Scalar beta = 1.0,
                       Scalar gamma = 0.0
)
{
  assert(A.rows() == B.rows());
  assert(A.rows() == C.rows());
  assert(A.cols() == B.cols());
  assert(A.cols() == C.cols());
  assert(A.values.size() == B.values.size());
  assert(A.values.size() == C.values.size());

  eigen::vector_map<Scalar> A1(const_cast<Scalar*>(A.values.data()), A.values.size());
  eigen::vector_map<Scalar> B1(const_cast<Scalar*>(B.values.data()), B.values.size());
  eigen::vector_map<Scalar> C1(const_cast<Scalar*>(C.values.data()), C.values.size());

  A1 = alpha * A1 + beta * B1 + gamma * C1;
  A.construct_csr();
}

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_SPARSE_MATRIX_H
