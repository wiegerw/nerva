// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/eigen.h
/// \brief add your file description here.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "nerva/neural_networks/matrix_operations.h"
#include "nerva/neural_networks/print_matrix.h"
#include "nerva/neural_networks/settings.h"
#include "nerva/utilities/text_utility.h"
#include "fmt/format.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <type_traits>

namespace nerva::eigen {

// Column Major = Fortran,
// Row Major = C
// Default Eigen: Column Major
// Default Numpy: Row Major

// Eigen matrices are by default stored in column-major order, meaning that the
// last index of the matrix changes the fastest. In other words, matrices are
// stored in a column-major order, which is the reverse of the default row-
// major order for NumPy arrays. This means that if you have a matrix, the
// elements are stored in a contiguous block of memory, with the elements of
// each column stored one after the other. In general, column-major order is
// the default storage format for many numerical libraries, including LAPACK
// and BLAS, as well as many high-level languages like Fortran.

#ifdef NERVA_COLWISE
static constexpr Eigen::StorageOptions default_matrix_layout = Eigen::ColMajor;
#else
static constexpr Eigen::StorageOptions default_matrix_layout = Eigen::RowMajor;
#endif

using vector = Eigen::Matrix<scalar, Eigen::Dynamic, 1>;
using matrix = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, default_matrix_layout>;

template <typename Scalar>
using vector_map = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;

template <typename Scalar>
using matrix_map = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, default_matrix_layout>>;

template <typename Scalar>
using vector_ref = Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;

template <typename Scalar>
using matrix_ref = Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, default_matrix_layout>>;

using sparse_matrix = Eigen::SparseMatrix<scalar>;

// Replace each coefficient x of the matrix X by x * (1 - x)
template <typename Matrix>
auto x_times_one_minus_x(const Matrix& X)
{
  return X.unaryExpr([](scalar t) { return t * (1-t); });
}

class eigen_slice
{
  private:
    std::vector<long>::const_iterator m_first;
    long m_size;

  public:
  eigen_slice(std::vector<long>::const_iterator first, long size)
        : m_first(first), m_size(size)
    {}

    [[nodiscard]] Eigen::Index size() const
    {
      return m_size;
    }

    Eigen::Index operator[] (Eigen::Index i) const
    {
      return *(m_first + i);
    }
};

inline
std::ostream& operator<<(std::ostream& out, const eigen_slice& s)
{
  for (auto i = 0; i < s.size(); i++)
  {
    out << s[i] << " ";
  }
  return out;
}

template <typename Scalar = scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> parse_vector(std::istream& from)
{
  long n;
  from >> n;

  if (!from)
  {
    throw std::runtime_error("Could not parse dimension");
  }

  std::string token;
  from >> token;
  if (!from || token != "#")
  {
    throw std::runtime_error("Could not parse separator");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x(n);
  for (auto i = 0; i < n; i++)
  {
    from >> x(i);
    if (!from)
    {
      throw std::runtime_error("Could not parse element " + std::to_string(i));
    }
  }

  return x;
}

template <typename Scalar = scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> parse_vector(const std::string& text)
{
  std::istringstream from(text);
  return parse_vector<Scalar>(from);
}

template <typename Scalar = scalar>
void load_vector(const std::string& filename, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& result)
{
  std::string text = read_text_fast(filename);
  result = parse_vector<Scalar>(text);
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> parse_matrix(std::istream& from)
{
  long m, n;
  from >> m >> n;

  if (!from)
  {
    throw std::runtime_error("Could not parse dimensions");
  }

  std::string token;
  from >> token;
  if (!from || token != "#")
  {
    throw std::runtime_error("Could not parse separator");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> x(m, n);
  for (auto i = 0; i < m; i++)
  {
    for (auto j = 0; j < n; j++)
    {
      from >> x(i, j);
      if (!from)
      {
        throw std::runtime_error("Could not parse element " + std::to_string(i) + "," + std::to_string(j));
      }
    }
  }

  return x;
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> parse_matrix(const std::string& text)
{
  std::istringstream from(text);
  return parse_matrix(from);
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void load_matrix(const std::string& filename, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& result)
{
  std::string text = read_text_fast(filename);
  result = parse_matrix(text);
}

template <typename Scalar = scalar>
void save_vector(const std::string& filename, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b)
{
  long n = b.size();
  std::ofstream to(filename);
  to << n << "#\n";
  for (long j = 0; j < n; j++)
  {
    if (j > 0)
    {
      to << ' ';
    }
    to << b(j);
  }
  to << '\n';
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void save_matrix(const std::string& filename, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  long m = A.rows();
  long n = A.cols();
  std::ofstream to(filename);
  to << m << ' ' << n << " #\n";
  for (long i = 0; i < m; i++)
  {
    for (long j = 0; j < n; j++)
    {
      if (j > 0)
      {
        to << ' ';
      }
      to << A(i, j);
    }
    to << '\n';
  }
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> standardize_column_data(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& x, scalar epsilon = 1e-20)
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> x_minus_mean = x.colwise() - x.rowwise().mean();
  auto stddev = x_minus_mean.array().square().rowwise().sum() / x.cols();
  return x_minus_mean.array().colwise() / (stddev + epsilon).sqrt();
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> standardize_row_data(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& x, scalar epsilon = 1e-20)
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> x_minus_mean = x.rowwise() - x.colwise().mean();
  auto stddev = x_minus_mean.array().square().colwise().sum() / x.rows();
  return x_minus_mean.array().rowwise() / (stddev + epsilon).sqrt();
}

// https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> sample_covariance(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& x)
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> x_minus_mean = x.rowwise() - x.colwise().mean();
  return (x_minus_mean.adjoint() * x_minus_mean) / double(x.rows() - 1);
}

// Performs the assignment A = B * C.
template <typename Matrix1, typename Matrix2, typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
inline
void assign_matrix_product(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A, const Matrix1& B, const Matrix2& C)
{
  A = B * C;
}

// Performs the assignment A = B * C. N.B. Only the existing entries of A are changed.
template <typename Matrix1, typename Matrix2>
inline
void assign_matrix_product(sparse_matrix& A, const Matrix1& B, const Matrix2& C)
{
  for (long j = 0; j < A.outerSize(); ++j)
  {
    for (sparse_matrix::InnerIterator it(A, j); it; ++it)
    {
      long i = it.row();
      it.valueRef() = B.row(i).dot(C.col(j));
    }
  }
}

// Assigns the value f() to each entry of the matrix
template <typename Function, typename Scalar = scalar>
void initialize_vector(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x, Function f)
{
  auto m = x.size();
  x = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::NullaryExpr(m, f);
}

// Assigns the value f() to each entry of the matrix
template <typename Function, typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void initialize_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A, Function f)
{
  auto m = A.rows();
  auto n = A.cols();
  A = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>::NullaryExpr(m, n, f);
}

// Assigns the value f() to each entry of the matrix
template <typename Function, typename Scalar = scalar>
void initialize_matrix(sparse_matrix& A, Function f)
{
  for (long j = 0; j < A.outerSize(); ++j)
  {
    for (sparse_matrix::InnerIterator it(A, j); it; ++it)
    {
      it.valueRef() = f();
    }
  }
}

// Assigns the value c to each entry of the matrix
template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void initialize_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A, scalar c)
{
  auto m = A.rows();
  auto n = A.cols();
  A = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>::Constant(m, n, c);
}

// Assigns the value c to each entry of the matrix
inline
void initialize_matrix(sparse_matrix& A, scalar c)
{
  for (long j = 0; j < A.outerSize(); ++j)
  {
    for (sparse_matrix::InnerIterator it(A, j); it; ++it)
    {
      it.valueRef() = c;
    }
  }
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
sparse_matrix to_sparse(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  typedef Eigen::Triplet<double> T;
  std::vector<T> triplets;

  auto m = A.rows();
  auto n = A.cols();

  for (auto i = 0; i < m; i++)
  {
    for (auto j = 0; j < n; j++)
    {
      auto value = A(i, j);
      if (value != scalar(0))
      {
        triplets.emplace_back(i, j, value);
      }
    }
  }

  sparse_matrix result(m, n);
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}

inline
void fill_matrix(sparse_matrix& A, scalar sparsity, std::mt19937& rng, scalar value = 0)
{
  typedef Eigen::Triplet<double> T;
  std::vector<T> triplets;

  auto m = A.rows();
  auto n = A.cols();

  std::bernoulli_distribution dist(scalar(1) - sparsity);
  for (auto i = 0; i < m; i++)
  {
    for (auto j = 0; j < n; j++)
    {
      if (dist(rng))
      {
        triplets.emplace_back(i, j, value);
      }
    }
  }

  A.setFromTriplets(triplets.begin(), triplets.end());
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void fill_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A, scalar sparsity, std::mt19937& rng, scalar value = 0)
{
  auto m = A.rows();
  auto n = A.cols();
  A = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>::Zero(m, n);

  std::bernoulli_distribution dist(scalar(1) - sparsity);
  for (auto i = 0; i < m; i++)
  {
    for (auto j = 0; j < n; j++)
    {
      if (dist(rng))
      {
        A(i, j) = value;
      }
    }
  }
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
void fill_matrix_random(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A, scalar density, scalar a, scalar b, std::mt19937& rng)
{
  auto m = A.rows();
  auto n = A.cols();
  A = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>::Zero(m, n);

  std::bernoulli_distribution dist(density);
  std::uniform_real_distribution<scalar> U(a, b);
  for (auto i = 0; i < m; i++)
  {
    for (auto j = 0; j < n; j++)
    {
      if (dist(rng))
      {
        A(i, j) = U(rng);
      }
    }
  }
}

template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
std::size_t nonzero_count(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  return (A.array() == 0.0).count();
}

template <typename Matrix>
std::size_t support_size(const Matrix& A)
{
  return (A.array() == 0).count();
}

template <typename Matrix>
std::size_t count_positive_elements(const Matrix& A)
{
  return (A.array() > 0).count();
}

template <typename Matrix>
std::size_t count_negative_elements(const Matrix& A)
{
  return (A.array() < 0).count();
}

// returns the L2 norm of (B - A)
template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
scalar l2_distance(const sparse_matrix& A, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& B, scalar epsilon)
{
  return (B - A).squaredNorm();
}

// Converts a target vector of integers into a one hot matrix.
// Every column will contain exactly one value 1.
template <typename Vector>
eigen::matrix to_one_hot_colwise(const Vector& T, long classes)
{
  long n = T.size();
  eigen::matrix result = eigen::matrix::Zero(classes, n);
  for (long i = 0; i < n; i++)
  {
    result(T(i), i) = scalar(1);
  }
  return result;
}

// Converts a target vector of integers into a one hot matrix.
// Every row will contain exactly one value 1.
template <typename Vector>
eigen::matrix to_one_hot_rowwise(const Vector& T, long classes)
{
  long n = T.size();
  eigen::matrix result = eigen::matrix::Zero(n, classes);
  for (long i = 0; i < n; i++)
  {
    result(i, T(i)) = scalar(1);
  }
  return result;
}

// Converts a one hot encoding X that was obtained from a long vector T to the original vector T.
// Every column of X should contain exactly one times the value 1.
inline
Eigen::Matrix<long, Eigen::Dynamic, 1> from_one_hot_colwise(const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>& X)
{
  auto m = X.rows();
  auto n = X.cols();
  Eigen::Matrix<long, Eigen::Dynamic, 1> T(n);

  for (auto j = 0; j < n; ++j)
  {
    for (auto i = 0; i < m; ++i)
    {
      if (X(i, j) == 1)
      {
        T(j) = i;
        break;
      }
    }
  }

  return T;
}

// Converts a one hot encoding X that was obtained from a long vector T to the original vector T.
// Every row of X should contain exactly one times the value 1.
inline
Eigen::Matrix<long, 1, Eigen::Dynamic> from_one_hot_rowwise(const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>& X)
{
  auto m = X.rows();
  auto n = X.cols();
  Eigen::Matrix<long, 1, Eigen::Dynamic> T(n);

  for (auto i = 0; i < m; ++i)
  {
    for (auto j = 0; j < n; ++j)
    {
      if (X(i, j) == 1)
      {
        T(i) = j;
        break;
      }
    }
  }

  return T;
}

template<class Matrix>
inline void write_dense_matrix(const std::string& filename, const Matrix& A)
{
  std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  if (out)
  {
    long rows = A.rows();
    long cols = A.cols();
    out.write(reinterpret_cast<char*>(&rows), sizeof(long));
    out.write(reinterpret_cast<char*>(&cols), sizeof(long));
    out.write(reinterpret_cast<const char*>(A.data()), rows * cols * sizeof(typename Matrix::Scalar));
    out.close();
  }
  else
  {
    std::cout << "Could not write to file: " << filename << std::endl;
  }
}

template<class Matrix>
inline void read_dense_matrix(const std::string& filename, Matrix& A)
{
  std::ifstream from(filename, std::ios::in | std::ios::binary);
  if (from)
  {
    long rows;
    long cols;
    from.read(reinterpret_cast<char*>(&rows), sizeof(long));
    from.read(reinterpret_cast<char*>(&cols), sizeof(long));
    A.resize(rows, cols);
    from.read(reinterpret_cast<char*>(A.data()), rows * cols * sizeof(typename Matrix::Scalar) );
    from.close();
  }
  else
  {
    std::cout << "Could not read from file: " << filename << std::endl;
  }
}

} // namespace nerva::eigen

