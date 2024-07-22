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
  eigen::vector y0 = Y.col(0);
  eigen::vector y1 = Y.col(1);
  eigen::vector y2 = Y.col(2);

  eigen::vector t0 = T.col(0);
  eigen::vector t1 = T.col(1);
  eigen::vector t2 = T.col(2);

  scalar L = loss(Y, T);
  scalar L0 = loss(y0, t0);
  scalar L1 = loss(y1, t1);
  scalar L2 = loss(y2, t2);
  eigen::matrix L012 {{ L0, L1, L2 }};

  std::cout << "L: " << L << std::endl;
  std::cout << "L0: " << L0 << std::endl;
  std::cout << "L1: " << L1 << std::endl;
  std::cout << "L2: " << L2 << std::endl;

  scalar epsilon = std::is_same<scalar, double>::value ? scalar(0.00001) : scalar(0.01);
  // CHECK((L - L012).squaredNorm() < epsilon);

  auto dL = loss.gradient(Y, T);
  auto dL0 = loss.gradient(y0, t0);
  auto dL1 = loss.gradient(y1, t1);
  auto dL2 = loss.gradient(y2, t2);

  scalar h = std::is_same<scalar, double>::value ? scalar(0.000001) : scalar(0.001);
  auto f0 = [&]() { return loss(y0, t0); };
  auto f1 = [&]() { return loss(y1, t1); };
  auto f2 = [&]() { return loss(y2, t2); };

  auto dL0_approx = approximate_derivative(f0, y0, h);
  auto dL1_approx = approximate_derivative(f1, y1, h);
  auto dL2_approx = approximate_derivative(f2, y2, h);

  std::cout << "\ndL:\n" << dL << std::endl;
  std::cout << "\ndL0:\n" << dL0 << std::endl;
  std::cout << "\ndL1:\n" << dL1 << std::endl;
  std::cout << "\ndL2:\n" << dL2 << std::endl;
  std::cout << "\ndL0_approx:\n" << dL0_approx << std::endl;
  std::cout << "\ndL1_approx:\n" << dL1_approx << std::endl;
  std::cout << "\ndL2_approx:\n" << dL2_approx << std::endl;

  CHECK((dL0 - dL0_approx).squaredNorm() < epsilon);
  CHECK((dL1 - dL1_approx).squaredNorm() < epsilon);
  CHECK((dL2 - dL2_approx).squaredNorm() < epsilon);

  CHECK((dL.col(0) - dL0).squaredNorm() < epsilon);
  CHECK((dL.col(1) - dL1).squaredNorm() < epsilon);
  CHECK((dL.col(2) - dL2).squaredNorm() < epsilon);
}

TEST_CASE("test_loss1")
{
  eigen::matrix Y {
    {1, 5, 3},
    {2, 1, 3}
  };

  eigen::matrix T {
    {1, 0, 0},
    {0, 1, 1}
  };

  test_loss("squared_error_loss", squared_error_loss(), 477.6448059082031, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 8.511181831359863, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -5.569567680358887, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -5.569567680358887, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.012376843020319939, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
    {1, 5, 3},
    {2, 1, 3},
    {4, 3, 7},
    {1, 5, 2}
  };

  eigen::matrix T {
    {1, 0, 0},
    {0, 0, 1},
    {0, 1, 0},
    {0, 0, 0}
  };

  test_loss("squared_error_loss", squared_error_loss(), 372.4724426269531, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 22.297447204589844, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -0.7536474466323853, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -0.7536474466323853, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.7411784529685974, Y, T);
}

