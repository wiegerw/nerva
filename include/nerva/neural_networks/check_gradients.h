// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/check_gradients.h
/// \brief add your file description here.

#pragma once

#include <iostream>
#include <iomanip>
#include "nerva/neural_networks/check_gradients.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/loss_functions_colwise.h"

namespace nerva {

inline
scalar relative_error(scalar x, scalar y)
{
  const scalar epsilon = 1e-9;
  return std::fabs(x - y) / (std::fabs(x) + epsilon);
}

inline
scalar safe_relative_error(scalar x, scalar y)
{
  const scalar epsilon = 1e-9;

  if (std::fabs(x) < 1e-7 && std::fabs(y) < 1e-7)
  {
    return scalar(0);
  }

  return std::fabs(x - y) / (std::fabs(x) + epsilon);
}

// N.B. This approximation is not ReLU friendly.
//template <typename Function>
//scalar approximate_derivative(Function f, scalar& x, scalar h)
//{
//  scalar x0 = x;
//  x = x0 + h;
//  scalar fplus = f();
//  x = x0 - h;
//  scalar fminus = f();
//  x = x0;
//  return (fplus - fminus) / (2*h);
//}

// This approximation is ReLU friendly
template <typename Function>
scalar approximate_derivative(Function f, scalar& x, scalar h)
{
  scalar x0 = x;
  scalar f0 = f();
  x = x0 + h;
  scalar fplus = f();
  x = x0;
  return (fplus - f0) / h;
}

template <typename Function>
eigen::vector approximate_derivative(Function f, eigen::vector& x, scalar h)
{
  long n = x.size();
  eigen::vector dx(n);
  for (long i = 0; i < n; i++)
  {
    dx[i] = approximate_derivative(f, x[i], h);
  }
  return dx;
}

template <typename Function>
eigen::matrix approximate_derivative(Function f, eigen::matrix& X, scalar h)
{
  long m = X.rows();
  long n = X.cols();
  eigen::matrix dX(m, n);
  for (long i = 0; i < m; i++)
  {
    for (long j = 0; j < n; j++)
    {
      dX(i, j) = approximate_derivative(f, X(i, j), h);
    }
  }
  return dX;
}

template <typename Function>
bool check_gradient(Function f, scalar& x, scalar dx, scalar h, unsigned int count = 5, scalar threshold = 0.05)
{
  for (unsigned int i = 0; i < count; i++)
  {
    if (safe_relative_error(dx, approximate_derivative(f, x, h)) < threshold)
    {
      return true;
    }
    h = h / 10;
  }
  return false;
}

template <typename Function>
void print_gradient(const std::string& name, Function f, scalar& x, scalar dx, scalar h, unsigned int count = 5)
{
  std::cout << name << " =" << std::setw(15) << dx;
  for (unsigned i = 0; i < count; i++)
  {
    std::cout << std::setw(15) << approximate_derivative(f, x, h);
    h = h / 10;
  }
  std::cout << std::endl;
}

// check gradient of a eigen::vector function
template <typename Function>
bool check_gradient(const std::string& name, Function f, eigen::vector& x, const eigen::vector& dx, scalar h, unsigned int count = 5, scalar threshold = 0.05)
{
  bool result = true;
  long n = x.size();
  for (long i = 0; i < n; i++)
  {
    if (!check_gradient(f, x[i], dx[i], h, count, threshold))
    {
      print_gradient(name + '[' + std::to_string(i) + ']', f, x[i], dx[i], h, count);
      result = false;
    }
  }
  return result;
}

// check gradient of an eigen::matrix function
template <typename Function>
bool check_gradient(const std::string& name, Function f, eigen::matrix& X, const eigen::matrix& dX, scalar h, unsigned int count = 5, scalar threshold = 0.05)
{
  bool result = true;
  auto n = X.rows();
  auto m = X.cols();
  for (auto i = 0; i < n; i++)
  {
    for (auto j = 0; j < m; j++)
    {
      if (!check_gradient(f, X(i, j), dX(i, j), h, count, threshold))
      {
        print_gradient(name + '[' + std::to_string(i) + ',' + std::to_string(j) + ']', f, X(i, j), dX(i, j), h, count);
        result = false;
      }
    }
  }
  return result;
}

// Check the gradient of an mkl::sparse_matrix_csr<scalar> function.
// Only the non-zero entries are checked.
template <typename Scalar, typename Function>
bool check_gradient(const std::string& name, Function f, mkl::sparse_matrix_csr<Scalar>& X, const mkl::sparse_matrix_csr<Scalar>& dX, Scalar h, unsigned int count = 5, Scalar threshold = 0.05)
{
  bool result = true;
  auto m = X.rows();
  auto X_values = const_cast<Scalar*>(X.values().data());
  const auto& X_col_index = X.col_index();
  const auto& X_row_index = X.row_index();
  const auto& dX_values = dX.values();

  for (long i = 0; i < m; i++)
  {
    for (long k = X_row_index[i]; k < X_row_index[i + 1]; k++)
    {
      long j = X_col_index[k];
      if (!check_gradient(f, X_values[k], dX_values[k], h, count, threshold))
      {
        print_gradient(name + '[' + std::to_string(i) + ',' + std::to_string(j) + ']', f, X_values[k], dX_values[k], h, count);
        result = false;
      }
    }
  }

  return result;
}

} // namespace nerva

