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

TEST_CASE("test_loss1")
{
  eigen::matrix Y {
          {5.774050024830831, 9.71037071742633, 9.126565598124968, 5.475153342545609},
          {5.681969245023855, 8.530089783461575, 1.822699552668345, 4.805757511235067},
          {4.239018833467587, 3.988667829565915, 3.083551447650138, 9.96952517674302}
  };

  eigen::matrix T {
          {0, 0, 0, 1},
          {0, 0, 0, 1},
          {0, 0, 0, 1}
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
          {0.6757806856333822, 4.264501534853461, 5.375741750827645, 9.268165527321143},
          {3.6768296513919405, 9.397688574768441, 7.715683430762863, 2.521534730815391},
          {1.2469074064358607, 0.7608470966516776, 7.857775979827943, 3.6492335526277095}
  };

  eigen::matrix T {
          {1, 0, 0, 0},
          {0, 0, 0, 1},
          {1, 0, 0, 0}
  };

  test_loss("squared_error_loss", squared_error_loss(), 372.4724426269531, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 22.297447204589844, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -0.7536474466323853, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -0.7536474466323853, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.7411784529685974, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
          {4.69019727804932, 5.512077572700557, 7.099039193062168, 9.95250032630546},
          {4.500814898177444, 8.318578433650654, 8.608042984219857, 9.432328470144075},
          {2.078325695323124, 3.8092099449870687, 9.574345266733257, 5.876624288834855}
  };

  eigen::matrix T {
          {0, 1, 0, 0},
          {0, 0, 0, 1},
          {0, 0, 1, 0}
  };

  test_loss("squared_error_loss", squared_error_loss(), 553.34765625, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.113748550415039, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -6.210171699523926, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -6.210171699523926, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.004179154988378286, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
          {3.9599045725261064, 3.102108414298299, 7.699659225449492, 8.739525149622137},
          {9.120160549855212, 4.9084471407520365, 8.453526188475614, 1.2784161679711974},
          {1.7002435652586707, 0.8969082286872924, 2.33111232688162, 8.488613335489225}
  };

  eigen::matrix T {
          {0, 0, 1, 0},
          {0, 0, 1, 0},
          {0, 1, 0, 0}
  };

  test_loss("squared_error_loss", squared_error_loss(), 391.4203186035156, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 10.037822723388672, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -4.066957950592041, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -4.066957950592041, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.3427145183086395, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
          {9.488208192808452, 0.8361163988866238, 6.222898432752989, 6.293502931619666},
          {7.7647597814242975, 7.807260374698173, 3.7651580770745503, 7.719384322881056},
          {4.853109172687492, 0.1442898395256863, 3.705933253579412, 0.10395986333454686}
  };

  eigen::matrix T {
          {1, 0, 0, 0},
          {0, 0, 1, 0},
          {1, 0, 0, 0}
  };

  test_loss("squared_error_loss", squared_error_loss(), 368.1732177734375, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.469424724578857, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -5.155459403991699, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -5.155459403991699, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.030749253928661346, Y, T);
}
