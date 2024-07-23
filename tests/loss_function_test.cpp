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
    {0.23759169, 0.42272727, 0.33968104},
    {0.43770149, 0.28115265, 0.28114586},
    {0.20141643, 0.45190243, 0.34668113},
    {0.35686849, 0.17944701, 0.46368450},
    {0.48552814, 0.26116029, 0.25331157},
  };

  eigen::matrix T {
    {1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 4.5487775802612305, Y, T);
}


//--- end generated code ---//
