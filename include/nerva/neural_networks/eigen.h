// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/eigen.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_EIGEN_H
#define NERVA_NEURAL_NETWORKS_EIGEN_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "nerva/neural_networks/scalar.h"
#include "nerva/utilities/text_utility.h"
#include "fmt/format.h"
#include <algorithm>
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

static constexpr Eigen::StorageOptions default_matrix_layout = Eigen::ColMajor;
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

template <typename Matrix>
void print_matrix(const std::string& name, const Matrix& x)
{
  std::cout << name << " =\n" << x << "\n";
}

template <typename Matrix>
void print_dimensions(const std::string& name, const Matrix& x)
{
  std::cout << name << " = " << x.rows() << " x " << x.cols() << std::endl;
}

template <typename Matrix>
void print_numpy_row_full(const Matrix& x, long i)
{
  long n = x.cols();
  std::cout << "   [";
  for (long j = 0; j < n; j++)
  {
    if (j > 0)
    {
      std::cout << ", ";
    }
    std::cout << fmt::format("{:11.8f}", x(i, j));
  }
  std::cout << "]\n";
}

template <typename Row>
void print_numpy_row(const Row& x, long edgeitems=3)
{
  using Scalar = typename Row::Scalar;

  auto print = [](auto x)
  {
    if constexpr (std::is_integral<Scalar>::value)
    {
      std::cout << fmt::format("{:3d}", x);
    }
    else
    {
      std::cout << fmt::format("{:11.8f}", x);
    }
  };

  long n = x.size();
  long left = n;
  long right = n;

  std::cout << "   [";

  if (n > 2*edgeitems)
  {
    left = std::min(n, edgeitems);
  }

  for (long j = 0; j < left; j++)
  {
    if (j > 0)
    {
      std::cout << ", ";
    }
    print(x(j));
  }

  if (n > 2*edgeitems)
  {
    std::cout << ",  ..., ";
    right = std::max(long(0), n - edgeitems);
  }

  for (long j = right; j < n; j++)
  {
    if (j > n - 3)
    {
      std::cout << ", ";
    }
    print(x(j));
  }

  std::cout << "]\n";
}

template <typename Vector>
void print_numpy_vector(const std::string& name, const Vector& x, long edgeitems=3)
{
  std::cout << name << "= " << x.size() << "\n";
  std::cout << std::setw(7);
  print_numpy_row(x, edgeitems);
}

template <typename Matrix>
struct matrix_row
{
  using Scalar = typename Matrix::Scalar;

  const Matrix& x;
  long i;

  matrix_row(const Matrix& x_, long i_)
    : x(x_), i(i_)
  {}

  auto operator()(long j) const
  {
    return x(i, j);
  }

  [[nodiscard]] long size() const
  {
    return x.cols();
  }
};

template <typename Matrix>
void print_numpy_matrix(const std::string& name, const Matrix& x, long edgeitems=3)
{
  std::cout << name << "= " << x.rows() << "x" << x.cols() << " norm = " << x.template lpNorm<Eigen::Infinity>() << "\n";
  long m = x.rows();
  long top = m;
  long bottom = m;

  if (m > 2*edgeitems)
  {
    top = std::min(m, edgeitems);
  }

  for (long i = 0; i < top; i++)
  {
    print_numpy_row(matrix_row(x, i), edgeitems);
  }

  if (m > 2*edgeitems)
  {
    std::cout << "   ...,\n";
    bottom = std::max(long(0), m - edgeitems);
  }

  for (long i = bottom; i < m; i++)
  {
    print_numpy_row(matrix_row(x, i), edgeitems);
  }
}

template <>
void print_numpy_matrix(const std::string& name, const sparse_matrix& x, long edgeitems)
{
  std::cout << name << "= " << x.rows() << "x" << x.cols() << "\n";
}

template <typename Matrix>
void print_cpp_matrix(const std::string& name, const Matrix& x)
{
  std::cout << name << " = " << x.rows() << "x" << x.cols() << "\n";
  std::cout << "{\n";
  for (long i = 0; i < x.rows(); i++)
  {
    std::cout << "  {";
    for (long j = 0; j < x.cols(); j++)
    {
      if (j != 0)
      {
        std::cout << ", ";
      }
      std::cout << x(i, j);
    }
    std::cout << (i < x.rows() - 1 ? "},\n" : "}\n");
  }
  std::cout << "}\n";
}

template <typename Scalar>
void print_cpp_matrix(const std::string& name, const Eigen::SparseMatrix<Scalar>& x)
{
  Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, default_matrix_layout> x1 = x;
  print_cpp_matrix(name, x1);
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

template <typename Scalar = scalar>
auto power_minus_half(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& Sigma, scalar epsilon = 0)
{
  return Sigma.unaryExpr([epsilon](scalar t) { return scalar(1) / std::sqrt(t + epsilon); });
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
long nonzero_count(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  return (A.array() == 0.0).count();
}

// returns the L2 norm of (B - A)
template <typename Scalar = scalar, int MatrixLayout = default_matrix_layout>
scalar l2_distance(const sparse_matrix& A, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& B, scalar epsilon)
{
  return (B - A).squaredNorm();
}

// Converts a target vector of longs into a one hot matrix.
// Every column will contain exactly one value 1.
inline
eigen::matrix to_one_hot(const Eigen::Matrix<long, Eigen::Dynamic, 1>& T, long classes)
{
  long n = T.size();
  eigen::matrix result = eigen::matrix::Zero(classes, n);
  for (long i = 0; i < n; i++)
  {
    result(T(i), i) = scalar(1);
  }
  return result;
}

// Converts a one hot encoding X that was obtained from a long vector T to the original vector T.
// Every column of X should contain exactly one times the value 1.
inline
Eigen::Matrix<long, Eigen::Dynamic, 1> from_one_hot(const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>& X)
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


} // namespace nerva::eigen

#endif // NERVA_NEURAL_NETWORKS_EIGEN_H
