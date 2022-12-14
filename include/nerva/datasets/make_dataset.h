// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/datasets/make_dataset.h
/// \brief add your file description here.

#ifndef NERVA_DATASETS_MAKE_DATASET_H
#define NERVA_DATASETS_MAKE_DATASET_H

#include "nerva/datasets/cifar10reader.h"
#include "nerva/datasets/dataset.h"

namespace nerva::datasets {

template<typename RandomNumberGenerator>
std::pair<eigen::matrix, eigen::matrix> make_dataset_chessboard(long n, RandomNumberGenerator rng)
{
  eigen::matrix X(2, n);
  eigen::matrix T(2, n);
  int N = 8;

  for (unsigned int i = 0; i < n; ++i)
  {
    scalar x = random_real(scalar(0.0), scalar(1.0), rng);
    scalar y = random_real(scalar(0.0), scalar(1.0), rng);
    int col = std::floor(x * N);
    int row = std::floor(y * N);
    bool is_dark = (row + col) % 2 == 0;
    X(0, i) = x;
    X(1, i) = y;
    T(0, i) = is_dark ? 1 : 0;
    T(1, i) = is_dark ? 0 : 1;
  }

  return {X, T};
}

template<typename RandomNumberGenerator>
std::pair<eigen::matrix, eigen::matrix> make_dataset_mini(long n, RandomNumberGenerator rng)
{
  long D = 3; // number of features
  long K = 2; // number of classes
  int a = 0;
  int b = 10;
  eigen::matrix X(D, n);
  eigen::matrix T(K, n);

  for (long i = 0; i < n; i++)
  {
    scalar x1 = random_integer(a, b, rng);
    scalar x2 = random_integer(a, b, rng);
    scalar x3 = random_integer(a, b, rng);
    eigen::vector x_i{{x1, x2, x3}};

    bool c = random_bool(rng);
    scalar t1 = c ? 1 : 0;
    scalar t2 = c ? 0 : 1;
    eigen::vector t_i{{t1, t2}};

    T.col(i) = t_i;
    X.col(i) = x_i;
  }

  return {X, T};
}

template<typename RandomNumberGenerator>
dataset make_dataset(const std::string& name, std::size_t n, RandomNumberGenerator rng)
{
  dataset result;
  auto n_train = n;
  auto n_test = n / 5;

  if (name == "chessboard")
  {
    std::tie(result.Xtrain, result.Ttrain) = make_dataset_chessboard(n_train, rng);
    std::tie(result.Xtest, result.Ttest) = make_dataset_chessboard(n_test, rng);
  }
  else if (name == "mini")
  {
    std::tie(result.Xtrain, result.Ttrain) = make_dataset_mini(n_train, rng);
    std::tie(result.Xtest, result.Ttest) = make_dataset_mini(n_test, rng);
  }
  else
  {
    throw std::runtime_error("Error: unknown dataset " + name);
  }
  return result;
}

inline
dataset load_cifar10_dataset(const std::string& directory)
{
  dataset result;
  cifar10reader reader;
  reader.read(directory);
  reader.normalize_data();
  std::tie(result.Xtrain, result.Ttrain, result.Xtest, result.Ttest) = reader.data();
  return result;
}

inline
dataset load_dataset(const std::string& directory)
{
  std::cout << "loading dataset from directory " << directory << std::endl;
  dataset result;
  eigen::load_matrix(directory + "/xtrain.txt", result.Xtrain);
  eigen::load_matrix(directory + "/ttrain.txt", result.Ttrain);
  eigen::load_matrix(directory + "/xvalid.txt", result.Xtest);
  eigen::load_matrix(directory + "/tvalid.txt", result.Ttest);
  std::cout << "done" << std::endl;
  return result;
}

} // namespace nerva::datasets

#endif // NERVA_DATASETS_MAKE_DATASET_H
