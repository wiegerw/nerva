// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_matrix.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MKL_MATRIX_H
#define NERVA_NEURAL_NETWORKS_MKL_MATRIX_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/print.h"
#include "nerva/utilities/random.h"
#include "nerva/utilities/stopwatch.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include <execution>
#include <vector>

namespace nerva::mkl {

// prototype declarations
template <typename Scalar> struct sparse_matrix_csr;
void compare_sizes(const mkl::sparse_matrix_csr<scalar>& A, const mkl::sparse_matrix_csr<scalar>& B);

// dense matrix
template <typename Scalar>
struct matrix
{
  int m; // number of rows
  int n; // number of columns
  Scalar* values;

  [[nodiscard]] int rows() const
  {
    return m;
  }

  [[nodiscard]] int cols() const
  {
    return n;
  }

  matrix(int m_, int n_, Scalar* values_)
   : m(m_), n(n_), values(values_)
  {}
};

// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-csr-matrix-storage-format.html
template <typename T>
struct sparse_matrix_csr
{
  using Scalar = T;

  std::vector<MKL_INT> row_index;
  std::vector<MKL_INT> columns;
  std::vector<T> values;
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

  sparse_status_t construct_csr()
  {
    // std::cout << "construct " << &csr << std::endl;
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
    return status;
  }

  void destruct_csr()
  {
    if (csr)
    {
      // std::cout << "destruct " << &csr << std::endl;
      mkl_sparse_destroy(csr); // TODO: How does MKL know that it should not destroy the row_index, columns and values arrays?
    }
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
   : m(rows), n(cols), row_index(std::move(row_index_)), columns(std::move(columns_)), values(std::move(values_))
  {
    construct_csr();
    check();
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

    if (auto status = construct_csr() != SPARSE_STATUS_SUCCESS)
    {
      throw std::runtime_error("mkl_sparse_?_create_csr reported status " + std::to_string(status));
    }
    check();
  }

  sparse_matrix_csr(const sparse_matrix_csr& A)
      : row_index(A.row_index),
        columns(A.columns),
        values(A.values),
        m(A.m),
        n(A.n)
  {
    construct_csr();
    check();
  }

  sparse_matrix_csr(sparse_matrix_csr&& A) noexcept
      : row_index(std::move(A.row_index)),
        columns(std::move(A.columns)),
        values(std::move(A.values)),
        m(A.m),
        n(A.n)
  {
    construct_csr();
    check();
  }

  sparse_matrix_csr& operator=(const sparse_matrix_csr& A)
  {
    row_index = A.row_index;
    columns = A.columns;
    values = A.values;
    m = A.m;
    n = A.n;
    destruct_csr();
    construct_csr();
    check();
    return *this;
  }

  sparse_matrix_csr& assign(const sparse_matrix_csr& A, T value = 0)
  {
    row_index = A.row_index;
    columns = A.columns;
    values = std::vector<T>(A.values.size(), value);
    m = A.m;
    n = A.n;
    destruct_csr();
    construct_csr();
    check();
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
  }

  [[nodiscard]] long rows() const
  {
    return m;
  }

  [[nodiscard]] long cols() const
  {
    return n;
  }

  void clear()
  {
    row_index.clear();
    columns.clear();
    values.clear();
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

  [[nodiscard]] scalar density() const
  {
    return scalar(values.size()) / (m * n);
  }
};

inline
void compare_sizes(const mkl::sparse_matrix_csr<scalar>& A, const mkl::sparse_matrix_csr<scalar>& B)
{
  if (A.rows() != B.rows() || A.cols() != B.cols())
  {
    throw std::runtime_error("matrix sizes do not match");
  }
}

template <typename Scalar>
void fill_matrix(sparse_matrix_csr<Scalar>& A, Scalar density, std::mt19937& rng, Scalar value = 0)
{
  long nonzero_count = std::lround(density * A.rows() * A.cols());
  A = sparse_matrix_csr<Scalar>(A.cols(), A.rows(), nonzero_count, rng, value);
}

template <typename Scalar, typename Function>
void initialize_matrix(sparse_matrix_csr<Scalar>& A, Function f)
{
  for (auto& value: A.values)
  {
    value = f();
  }
}

template <typename Scalar>
void load_matrix(const std::string& filename, sparse_matrix_csr<Scalar>& A)
{
  auto m = A.rows();
  auto n = A.cols();
  eigen::matrix A1(m, n);
  eigen::load_matrix(filename, A1);

  long count = 0;
  for (long i = 0; i < m; i++)
  {
    for (long k = A.row_index[i]; k < A.row_index[i + 1]; k++)
    {
      long j = A.columns[count];
      A.values[count] = A1(i, j);
      count++;
    }
  }
}

// Performs the assignment A := B, with A, B sparse. A and B must have the same support.
template <typename Scalar>
void assign_matrix(sparse_matrix_csr<Scalar>& A, const sparse_matrix_csr<Scalar>& B)
{
  std::copy(B.values.begin(), B.values.end(), A.values.begin());
}

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_MATRIX_H
