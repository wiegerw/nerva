// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file matrix_operation_test.cpp
/// \brief Tests for matrix operations.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/eigen.h"
#include <iostream>

using namespace nerva;

TEST_CASE("test_colwise_replicate")
{
  eigen::matrix A {
    {1, 2, 3}
  };

  eigen::matrix B = eigen::colwise_replicate(A, 2);

  eigen::matrix C {
    {1, 2, 3},
    {1, 2, 3}
  };

  CHECK_EQ(A.rows(), 1);
  CHECK_EQ(B, C);
}
