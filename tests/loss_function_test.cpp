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
    {0.37123377, 0.22913980, 0.52343031, 0.32794611, 0.38442905},
    {0.32155039, 0.38546351, 0.22596862, 0.33606186, 0.28971702},
    {0.30721585, 0.38539670, 0.25060107, 0.33599203, 0.32585394},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000},
    {1.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.700765371322632, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.651103973388672, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 6.038372993469238, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 6.038372993469238, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.7645041942596436, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
    {0.26507374, 0.33868876, 0.32442389, 0.29234829, 0.44228604},
    {0.35041776, 0.32097087, 0.42761531, 0.37522641, 0.24338417},
    {0.38450850, 0.34034037, 0.24796079, 0.33242529, 0.31432980},
  };

  eigen::matrix T {
    {1.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.833115816116333, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.727041721343994, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 6.245758533477783, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 6.245759010314941, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.797585964202881, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
    {0.42619403, 0.34836590, 0.37067701, 0.38766850, 0.33742142},
    {0.23064549, 0.33952078, 0.34860632, 0.36117711, 0.44983957},
    {0.34316048, 0.31211332, 0.28071667, 0.25115439, 0.21273900},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 1.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000},
    {1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 2.9479236602783203, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.279299736022949, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 4.888670444488525, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 4.888670921325684, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.6101226806640625, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
    {0.23061659, 0.42460632, 0.43308898, 0.53847172, 0.26530370},
    {0.29721665, 0.39750320, 0.30700550, 0.23244817, 0.33082264},
    {0.47216676, 0.17789048, 0.25990553, 0.22908011, 0.40387366},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 1.00000000, 0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.9202888011932373, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.734646320343018, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 6.458463191986084, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 6.458463668823242, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.79862904548645, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
    {0.37433283, 0.45234160, 0.22061235, 0.18245918, 0.23087405},
    {0.37087153, 0.24802873, 0.28949737, 0.40047895, 0.28281153},
    {0.25479564, 0.29962967, 0.48989029, 0.41706187, 0.48631441},
  };

  eigen::matrix T {
    {0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.2525599002838135, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.405562400817871, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.344449520111084, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.344449996948242, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.6603212356567383, Y, T);
}


//--- end generated code ---//
