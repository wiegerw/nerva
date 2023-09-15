// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/datasets/dataset.h
/// \brief add your file description here.

#pragma once

#include "nerva/datasets/cifar10reader.h"
#include "nerva/datasets/dataset.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/numpy_eigen.h"
#include "nerva/utilities/random.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>
#include <string>
#include <random>

namespace nerva::datasets {

using matrix_view = eigen::matrix_map<scalar>;
using matrix_ref = eigen::matrix_ref<scalar>;
using long_vector = Eigen::Matrix<long, Eigen::Dynamic, 1>;

enum class dataset_orientation {
  colwise,
  rowwise
};

template <typename Matrix>
void dataset_info(const Matrix& Xtrain, const Matrix& Ttrain, const Matrix& Xtest, const Matrix& Ttest, dataset_orientation orientation)
{
  std::cout << "dataset orientation = " << (orientation == dataset_orientation::colwise ? "colwise" : "rowwise") << std::endl;
  if (orientation == dataset_orientation::colwise)
  {
    print_numpy_matrix("Xtrain", Xtrain.transpose());
    print_numpy_matrix("Ttrain", Ttrain.transpose());
    print_numpy_matrix("Xtest", Xtest.transpose());
    print_numpy_matrix("Ttest", Ttest.transpose());
  }
  else
  {
    print_numpy_matrix("Xtrain", Xtrain);
    print_numpy_matrix("Ttrain", Ttrain);
    print_numpy_matrix("Xtest", Xtest);
    print_numpy_matrix("Ttest", Ttest);
  }
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
template <typename Matrix>
void dataset_load(const std::string& filename, Matrix& Xtrain, Matrix& Ttrain, Matrix& Xtest, Matrix& Ttest, dataset_orientation orientation)
{
  std::cout << "Loading dataset from file " << filename << std::endl;

  if (!std::filesystem::exists(std::filesystem::path(filename)))
  {
    throw std::runtime_error("Could not load file '" + filename + "'");
  }

  pybind11::dict data = pybind11::module::import("numpy").attr("load")(filename);

  if (orientation == dataset_orientation::colwise)
  {
    Xtrain = eigen::extract_matrix<scalar>(data, "Xtrain").transpose();
    Xtest = eigen::extract_matrix<scalar>(data, "Xtest").transpose();
    auto Ttrain_ = eigen::extract_column_vector<long>(data, "Ttrain");
    auto Ttest_ = eigen::extract_column_vector<long>(data, "Ttest");
    long num_classes = Ttrain_.maxCoeff() + 1;
    Ttrain = eigen::to_one_hot_colwise(Ttrain_, num_classes);
    Ttest = eigen::to_one_hot_colwise(Ttest_, num_classes);
  }
  else
  {
    Xtrain = eigen::extract_matrix<scalar>(data, "Xtrain");
    Xtest = eigen::extract_matrix<scalar>(data, "Xtest");
    auto Ttrain_ = eigen::extract_row_vector<long>(data, "Ttrain");
    auto Ttest_ = eigen::extract_row_vector<long>(data, "Ttest");
    long num_classes = Ttrain_.maxCoeff() + 1;
    Ttrain = eigen::to_one_hot_rowwise(Ttrain_, num_classes);
    Ttest = eigen::to_one_hot_rowwise(Ttest_, num_classes);
  }
}

template <typename Matrix>
void dataset_save(const std::string& filename, const Matrix& Xtrain, const Matrix& Ttrain, const Matrix& Xtest, const Matrix& Ttest, dataset_orientation orientation)
{
  std::cout << "Saving dataset to file " << filename << std::endl;

  eigen::matrix Xtrain_ = Xtrain.transpose();
  eigen::matrix Xtest_ = Xtest.transpose();

  long_vector Ttrain_ = eigen::from_one_hot_colwise(Ttrain);
  long_vector Ttest_ = eigen::from_one_hot_colwise(Ttest);

  pybind11::dict data;
  data["Xtrain"] = pybind11::array_t<scalar, pybind11::array::f_style>({Xtrain_.rows(), Xtrain_.cols()}, Xtrain_.data());
  data["Ttrain"] = pybind11::array_t<long, pybind11::array::f_style>(Ttrain_.size(), Ttrain_.data());
  data["Xtest"] = pybind11::array_t<scalar, pybind11::array::f_style>({Xtest_.rows(), Xtest_.cols()}, Xtest_.data());
  data["Ttest"] = pybind11::array_t<long, pybind11::array::f_style>(Ttest_.size(), Ttest_.data());

  pybind11::module::import("numpy").attr("savez_compressed")(filename, **data);
}

struct dataset
{
  eigen::matrix Xtrain;
  eigen::matrix Ttrain;
  eigen::matrix Xtest;
  eigen::matrix Ttest;
  dataset_orientation orientation = dataset_orientation::colwise;

  explicit dataset(dataset_orientation orientation_=dataset_orientation::colwise)
   : orientation(orientation_)
  {}

  dataset(eigen::matrix  Xtrain_,
          const long_vector& Ttrain_,
          eigen::matrix  Xtest_,
          const long_vector& Ttest_,
          dataset_orientation orientation_=dataset_orientation::colwise

  )
   : Xtrain(std::move(Xtrain_)), Xtest(std::move(Xtest_)), orientation(orientation_)
  {
    long num_classes = Ttrain_.maxCoeff() + 1;
    if (orientation_ == dataset_orientation::colwise)
    {
      Ttrain = eigen::to_one_hot_colwise(Ttrain_, num_classes);
      Ttest = eigen::to_one_hot_colwise(Ttest_, num_classes);
    }
    else
    {
      Ttrain = eigen::to_one_hot_rowwise(Ttrain_, num_classes);
      Ttest = eigen::to_one_hot_rowwise(Ttest_, num_classes);
    }
  }

  void info() const
  {
    dataset_info(Xtrain, Ttrain, Xtest, Ttest, orientation);
  }

  // Precondition: the python interpreter must be running.
  // This can be enforced using `py::scoped_interpreter guard{};`
  void load(const std::string& filename)
  {
    dataset_load(filename, Xtrain, Ttrain, Xtest, Ttest, orientation);
  }

  void save(const std::string& filename) const
  {
    dataset_save(filename, Xtrain, Ttrain, Xtest, Ttest, orientation);
  }

  void transpose()
  {
    Xtrain.transposeInPlace();
    Xtest.transposeInPlace();
    Ttrain.transposeInPlace();
    Ttest.transposeInPlace();
  }
};

// contains references to matrices
struct dataset_view
{
  matrix_ref Xtrain;
  matrix_ref Ttrain;
  matrix_ref Xtest;
  matrix_ref Ttest;
  dataset_orientation orientation;

  dataset_view(const matrix_ref& Xtrain_view,
               const matrix_ref& Ttrain_view,
               const matrix_ref& Xtest_view,
               const matrix_ref& Ttest_view,
               dataset_orientation orientation_=dataset_orientation::colwise
  )
    : Xtrain(Xtrain_view),
      Ttrain(Ttrain_view),
      Xtest(Xtest_view),
      Ttest(Ttest_view),
      orientation(orientation_)
  {}

  void info() const
  {
    dataset_info(Xtrain, Ttrain, Xtest, Ttest, orientation);
  }

  // Precondition: the python interpreter must be running.
  // This can be enforced using `py::scoped_interpreter guard{};`
  void load(const std::string& filename)
  {
    dataset_load(filename, Xtrain, Ttrain, Xtest, Ttest, orientation);
  }

  void save(const std::string& filename) const
  {
    dataset_save(filename, Xtrain, Ttrain, Xtest, Ttest, orientation);
  }
};

inline
matrix_view make_matrix_view(eigen::matrix& X)
{
  return {X.data(), X.rows(), X.cols()};
}

inline dataset_view make_dataset_view(dataset& data)
{
  return {make_matrix_view(data.Xtrain),
          make_matrix_view(data.Ttrain),
          make_matrix_view(data.Xtest),
          make_matrix_view(data.Ttest),
          data.orientation
         };
}

} // namespace nerva

