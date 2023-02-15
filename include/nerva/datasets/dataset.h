// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/datasets/dataset.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_DATASET_H
#define NERVA_NEURAL_NETWORKS_DATASET_H

#include "nerva/datasets/cifar10reader.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/numpy_eigen.h"
#include "nerva/utilities/random.h"
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

struct dataset
{
  using long_vector = Eigen::Matrix<long, Eigen::Dynamic, 1>;

  eigen::matrix Xtrain;
  eigen::matrix Ttrain;
  eigen::matrix Xtest;
  eigen::matrix Ttest;

  dataset() = default;

  dataset(eigen::matrix  Xtrain_,
          const long_vector& Ttrain_,
          eigen::matrix  Xtest_,
          const long_vector& Ttest_
  )
   : Xtrain(std::move(Xtrain_)), Xtest(std::move(Xtest_))
  {
    eigen::to_one_hot(Ttrain_, Ttrain);
    eigen::to_one_hot(Ttest_, Ttest);
  }

  void info() const
  {
    eigen::print_numpy_matrix("Xtrain", Xtrain);
    eigen::print_numpy_matrix("Ttrain", Ttrain);
    eigen::print_numpy_matrix("Xtest", Xtest);
    eigen::print_numpy_matrix("Ttest", Ttest);
  }

  // Precondition: the python interpreter must be running.
  // This can be enforced using `py::scoped_interpreter guard{};`
  void import_cifar10_from_npz(const std::string& filename)
  {
    std::cout << "Loading data from file " << filename << std::endl;

    if (!std::filesystem::exists(std::filesystem::path(filename)))
    {
      throw std::runtime_error("Could not load file '" + filename + "'");
    }

    pybind11::dict data = pybind11::module::import("numpy").attr("load")(filename);
    Xtrain = eigen::extract_matrix<scalar>(data, "Xtrain").transpose();
    Xtest = eigen::extract_matrix<scalar>(data, "Xtest").transpose();
    auto Ttrain_ = eigen::extract_vector<long>(data, "Ttrain");
    auto Ttest_ = eigen::extract_vector<long>(data, "Ttest");
    eigen::to_one_hot(Ttrain_, Ttrain);
    eigen::to_one_hot(Ttest_, Ttest);
  }
};

// contains references to matrices
struct dataset_view
{
  matrix_ref Xtrain;
  matrix_ref Ttrain;
  matrix_ref Xtest;
  matrix_ref Ttest;

  dataset_view(const matrix_ref& Xtrain_view,
               const matrix_ref& Ttrain_view,
               const matrix_ref& Xtest_view,
               const matrix_ref& Ttest_view
  )
    : Xtrain(Xtrain_view),
      Ttrain(Ttrain_view),
      Xtest(Xtest_view),
      Ttest(Ttest_view)
  {}

  void info() const
  {
    eigen::print_numpy_matrix("Xtrain", Xtrain);
    eigen::print_numpy_matrix("Ttrain", Ttrain);
    eigen::print_numpy_matrix("Xtest", Xtest);
    eigen::print_numpy_matrix("Ttest", Ttest);
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
          make_matrix_view(data.Ttest)
         };
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_DATASET_H
