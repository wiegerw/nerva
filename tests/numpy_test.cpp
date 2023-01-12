// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file numpy_test.cpp
/// \brief Tests for integration with Numpy.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

template <typename Scalar, int MatrixLayout>
py::array_t<Scalar> to_numpy(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>;

  // Create an empty array result of the correct size
  py::array_t<Scalar> result({A.rows(), A.cols()});

  // Copy A to result
  Eigen::Map<matrix_t>(result.mutable_data(), A.rows(), A.cols()) = A;

  return result;
}

template <typename Scalar = double, int MatrixLayout = Eigen::ColMajor>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> from_numpy(const py::array_t<double>& A)
{
  using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>;

  assert(A.ndim() == 2);
  auto shape = A.shape();
  return Eigen::Map<matrix_t>(const_cast<Scalar*>(A.data()), shape[0], shape[1]);
}

// Save a Numpy array to a file in .npy format
template <typename Scalar>
void save_numpy_array(const std::string& filename, const py::array_t<Scalar>& A)
{
  py::module::import("numpy").attr("save")(filename, A);
}

template <typename Scalar = double>
py::array_t<Scalar> load_numpy_array(const std::string& filename)
{
  return py::module::import("numpy").attr("load")(filename);
}

// compare two dimensional Numpy arrays
template <typename Scalar>
bool compare_numpy_arrays(const py::array_t<Scalar>& x, const py::array_t<Scalar>& y)
{
  return std::equal(x.data(), x.data() + x.size(), y.data(), y.data() + y.size());
}

template <typename Scalar>
void print_numpy_array(const py::array_t<Scalar>& x)
{
  auto n = x.size();
  const double* data = x.data();
  for (auto i = 0; i < n; i++)
  {
    std::cout << data[i] << ' ';
  }
  std::cout << std::endl;
}

TEST_CASE("test_to_numpy")
{
  py::scoped_interpreter guard{};  // Initialize the interpreter
  py::module numpy = py::module::import("numpy");

  Eigen::MatrixXd A {
    {1, 2, 3, 4},
    {5, 6, 7, 8}
  };

  py::array_t<double> B = to_numpy(A);
  print_numpy_array(B);

  auto D = from_numpy(B);
  CHECK(A == D);

  save_numpy_array("B.npy", B);

  py::array_t<double> C = load_numpy_array("B.npy");
  CHECK(compare_numpy_arrays(B, C));
}
