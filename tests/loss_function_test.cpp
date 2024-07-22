// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file loss_function_test.cpp
/// \brief Tests for loss functions.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/check_gradients.h"
#include "nerva/neural_networks/loss_functions.h"
#include <iostream>
#include <type_traits>

using namespace nerva;

template <typename LossFunction>
void test_loss(const std::string& name, LossFunction loss, scalar expected, const eigen::matrix& Y, const eigen::matrix& T)
{
  std::cout << "\n=== test_loss " << name << " ===" << std::endl;
  scalar L = loss(Y, T);
  scalar epsilon = std::is_same<scalar, double>::value ? scalar(0.000001) : scalar(0.001);
  CHECK(expected == doctest::Approx(L).epsilon(epsilon));
}

//--- begin generated code ---//
TEST_CASE("test_loss1")
{
  eigen::matrix Y {
    {0.37454074, 0.15601948, 0.60111541},
    {0.95071436, 0.15599536, 0.70807287},
    {0.73199421, 0.05808455, 0.02058547},
    {0.59865889, 0.86617628, 0.96990988},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 1.00000000},
    {0.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -1.4627270698547363, Y, T);
}


//--- end generated code ---//
