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
#include "nerva/neural_networks/matrix.h"
#include "nerva/neural_networks/softmax_functions_colwise.h"
#include "nerva/neural_networks/softmax_functions_rowwise.h"
#include <iostream>

using namespace nerva;

TEST_CASE("test_softmax")
{
  std::cout << "\n=== test_softmax ===" << std::endl;

  eigen::matrix X {
    {1.0, 2.0, 7.0},
    {3.0, 4.0, 9.0}
  };

  eigen::matrix Y = stable_softmax()(X);

  eigen::matrix x1 = X.col(0);
  eigen::matrix x2 = X.col(1);
  eigen::matrix x3 = X.col(2);

  eigen::matrix y1 = stable_softmax()(x1);
  eigen::matrix y2 = stable_softmax()(x2);
  eigen::matrix y3 = stable_softmax()(x3);

  eigen::print_numpy_matrix("Y", Y);
  eigen::print_numpy_matrix("y1", y1);
  eigen::print_numpy_matrix("y2", y2);
  eigen::print_numpy_matrix("y3", y3);

  CHECK_EQ(y1, Y.col(0));
  CHECK_EQ(y2, Y.col(1));
  CHECK_EQ(y3, Y.col(2));
}

template <typename Matrix>
void check_close(const Matrix& Xc, const Matrix& Xr)
{
  scalar tolerance = 1e-6;
  Matrix Yc = Xr.transpose();

  print_numpy_matrix("Xc", Xc);
  print_numpy_matrix("Yc", Yc);

  REQUIRE(Xc.isApprox(Yc, tolerance));
}

TEST_CASE("test_softmax_rowwise_colwise")
{
  std::cout << "\n=== test_softmax ===" << std::endl;

  eigen::matrix Xc {
    {1.0, 2.0, 7.0},
    {3.0, 4.0, 9.0}
  };

  eigen::matrix Xr = Xc.transpose();
  check_close(colwise::softmax()(Xc), rowwise::softmax()(Xr));
  check_close(colwise::stable_softmax()(Xc), rowwise::stable_softmax()(Xr));
  check_close(colwise::log_softmax()(Xc), rowwise::log_softmax()(Xr));
  check_close(colwise::stable_log_softmax()(Xc), rowwise::stable_log_softmax()(Xr));
}
