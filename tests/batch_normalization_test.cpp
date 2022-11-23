// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file batch_normalization_test.cpp
/// \brief Tests for batch normalization.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/eigen.h"
#include <iostream>

using namespace nerva;

inline
eigen::matrix standardize_column_data1(const eigen::matrix& x, double epsilon = 1e-20)
{
  eigen::matrix R = x.colwise() - x.rowwise().mean();
  auto Sigma = R.array().square().rowwise().sum() / x.cols();
  return R.array().colwise() / (Sigma + epsilon).sqrt();
}

inline
eigen::matrix standardize_column_data2(const eigen::matrix& x, double epsilon = 1e-20)
{
  eigen::matrix R = x.colwise() - x.rowwise().mean();
  auto Sigma = (R * R.transpose()).diagonal().array() / x.cols();
  return R.array().colwise() / (Sigma + epsilon).sqrt();
}

// Compare two different computations of the standardization of data, as it
// is used in batch normalization.
TEST_CASE("test_standardize")
{
  std::cout << "test_standardize" << std::endl;

  eigen::matrix x {
    {1, 2, 3},
    {3, 7, 2},
  };

  eigen::matrix expected {
    {-1.22474487, 0.        ,  1.22474487},
    {-0.46291005, 1.38873015, -0.9258201},
  };

  eigen::matrix y1 = standardize_column_data1(x);
  std::cout << "y1=\n" << y1 << std::endl;
  CHECK_LT((expected - y1).squaredNorm(), 1e-10);

  eigen::matrix y2 = standardize_column_data2(x);
  std::cout << "y2=\n" << y2 << std::endl;
  CHECK_LT((expected - y2).squaredNorm(), 1e-10);
}
