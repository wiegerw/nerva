// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file softmax_test.cpp
/// \brief Tests for gradients.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/check_gradients.h"
#include "nerva/neural_networks/layers.h"

#include <cmath>

using namespace nerva;

TEST_CASE("test_softmax_colwise")
{
  eigen::matrix X {
    {6, 8, 10},
    {3, 12, 15},
  };

  eigen::matrix softmax_X = stable_softmax_colwise()(X);
  eigen::matrix log_softmax_X = stable_log_softmax_colwise()(X);
  eigen::matrix log_softmax_X1 = softmax_X.unaryExpr([](auto t) { return std::log(t); });

  print_numpy_matrix("X", X);
  print_numpy_matrix("softmax_X", softmax_X);
  print_numpy_matrix("log_softmax_X", log_softmax_X);
  print_numpy_matrix("log_softmax_X1", log_softmax_X1);

  CHECK_LE((log_softmax_X - log_softmax_X1).squaredNorm(), 1e-10);
}

TEST_CASE("test_softmax_rowwise")
{
  eigen::matrix X {
    {6, 8, 10},
    {3, 12, 15},
  };

  eigen::matrix softmax_X = stable_softmax_rowwise()(X);
  eigen::matrix log_softmax_X = stable_log_softmax_rowwise()(X);
  eigen::matrix log_softmax_X1 = softmax_X.unaryExpr([](auto t) { return std::log(t); });

  print_numpy_matrix("X", X);
  print_numpy_matrix("softmax_X", softmax_X);
  print_numpy_matrix("log_softmax_X", log_softmax_X);
  print_numpy_matrix("log_softmax_X1", log_softmax_X1);

  CHECK_LE((log_softmax_X - log_softmax_X1).squaredNorm(), 1e-10);
}
