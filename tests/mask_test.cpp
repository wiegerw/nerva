// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mask_test.cpp
/// \brief Tests for masking.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/masking.h"
#include <Eigen/Dense>
#include <iostream>

TEST_CASE("test_masking")
{
  Eigen::MatrixXf mat(3, 3);
  mat << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

  std::cout << "Original matrix:\n" << mat << std::endl;

  Eigen::MatrixXf pattern(3, 3);
  pattern << 1, 0, 0,
             1, 0, 0,
             1, 1, 1;

  Eigen::MatrixXf expected(3, 3);
  expected << 1, 0, 0,
              4, 0, 0,
              7, 8, 9;

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask = nerva::create_binary_mask(pattern);
  std::cout << "mask:\n" << mask << std::endl;

  nerva::apply_binary_mask(mat, mask);

  std::cout << "Matrix with masked elements set to zero:\n" << mat << std::endl;

  CHECK_EQ(mat, expected);
}
