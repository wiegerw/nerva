// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file softmax_test.cpp
/// \brief Tests for softmax functions.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/softmax_functions.h"
#include <iostream>

using namespace nerva;

void test_stable_softmax(const eigen::matrix& X)
{
  eigen::matrix Y = stable_softmax()(X);

  auto N = X.rows();
  for (auto i = 0; i < N; i++)
  {
    eigen::matrix x_i = X.row(i);
    eigen::matrix y_i = stable_softmax()(x_i);
    CHECK_EQ(y_i, Y.row(i));
  }
}

TEST_CASE("test_softmax")
{
  std::cout << "\n=== test_softmax ===" << std::endl;

  eigen::matrix X1 {
    {1.0, 2.0, 7.0},
    {3.0, 4.0, 9.0}
  };
  test_stable_softmax(X1);

  eigen::matrix X2 {
    {1.0, 2.0},
    {2.0, 4.0},
    {6.0, 2.0}
  };
  test_stable_softmax(X2);
}
