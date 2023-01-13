// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/numpy_eigen.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_NUMPY_EIGEN_H
#define NERVA_NEURAL_NETWORKS_NUMPY_EIGEN_H

#include "nerva/neural_networks/eigen.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace nerva::eigen {

namespace py = pybind11;

template <typename Scalar, int Rows, int Cols, int MatrixLayout>
py::array_t<Scalar> to_numpy(const Eigen::Matrix<Scalar, Rows, Cols, MatrixLayout>& A)
{
  using matrix_t = Eigen::Matrix<Scalar, Rows, Cols, MatrixLayout>;

  // Create an empty array result of the correct size
  py::array_t<Scalar> result({A.rows(), A.cols()});

  // Copy A to result
  Eigen::Map<matrix_t>(result.mutable_data(), A.rows(), A.cols()) = A;

  return result;
}

template <typename Scalar = double, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic, int MatrixLayout = Eigen::ColMajor>
Eigen::Matrix<Scalar, Rows, Cols, MatrixLayout> from_numpy(const py::array_t<Scalar>& A)
{
  using matrix_t = Eigen::Matrix<Scalar, Rows, Cols, MatrixLayout>;

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

} // namespace nerva::eigen

#endif // NERVA_NEURAL_NETWORKS_NUMPY_EIGEN_H
