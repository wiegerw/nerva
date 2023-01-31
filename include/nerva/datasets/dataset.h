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
#include <vector>
#include <string>
#include <random>

namespace nerva::datasets {

using matrix_view = eigen::matrix_map<scalar>;
using matrix_ref = eigen::matrix_ref<scalar>;

struct dataset
{
  eigen::matrix Xtrain;
  eigen::matrix Ttrain;
  eigen::matrix Xtest;
  eigen::matrix Ttest;

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
    namespace py = pybind11;
    auto np = py::module::import("numpy");
    auto io = py::module::import("io");

    if (!std::filesystem::exists(std::filesystem::path(filename)))
    {
      throw std::runtime_error("Could not load file '" + filename + "'");
    }

    py::dict d = np.attr("load")(filename);

    Xtrain = nerva::eigen::from_numpy(d["Xtrain"].cast<py::array_t<scalar>>());

    // create one hot encoded matrix Ttrain
    Ttrain = eigen::matrix::Zero(10, 50000);
    auto ttrain = d["Ttrain"].cast<py::array_t<long>>();
    auto rtrain = ttrain.unchecked<1>();
    for (long i = 0; i < 50000; i++)
    {
      Ttrain(rtrain(i), i) = scalar(1);
    }

    Xtest  = nerva::eigen::from_numpy(d["Xtest"].cast<py::array_t<scalar>>());

    // create one hot encoded matrix Ttest
    Ttest = eigen::matrix::Zero(10, 10000);
    auto ttest = d["Ttest"].cast<py::array_t<long>>();
    auto rtest = ttest.unchecked<1>();
    for (long i = 0; i < 10000; i++)
    {
      Ttest(rtest(i), i) = scalar(1);
    }
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
