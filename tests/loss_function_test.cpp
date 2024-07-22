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
          {3.74540181, 9.50714311, 7.31993969, 5.98658524},
          {1.56018725, 1.55994605, 0.58083706, 8.66176159},
          {6.01115052, 7.08072607, 0.20584592, 9.69909855},
  };

  eigen::matrix T {
          {0.00000000, 0.00000000, 0.00000000, 1.00000000},
          {0.00000000, 1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 428.19659423828125, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 13.471577644348145, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -4.191548824310303, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -4.191548824310303, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.1940920650959015, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
          {0.00778866, 9.92211560, 6.17481548, 6.11653199},
          {0.07066405, 0.23062523, 5.24774708, 3.99861032},
          {0.46665759, 9.73755521, 2.32771417, 0.90606525},
  };

  eigen::matrix T {
          {0.00000000, 1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 0.00000000, 1.00000000, 0.00000000},
          {0.00000000, 0.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 289.69976806640625, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 9.138801574707031, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -3.8539209365844727, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -3.8539209365844727, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.3446991443634033, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
          {5.14234487, 5.92414610, 0.46450508, 6.07544891},
          {1.70524207, 0.65051686, 9.48885542, 9.65632037},
          {8.08397367, 3.04613839, 0.97672204, 6.84233058},
  };

  eigen::matrix T {
          {0.00000000, 0.00000000, 1.00000000, 0.00000000},
          {0.00000000, 0.00000000, 0.00000000, 1.00000000},
          {0.00000000, 0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 388.4859619140625, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 14.40466594696045, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -1.4772768020629883, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -1.4772768020629883, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.8072657585144043, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
          {6.09996697, 8.33194928, 1.73364736, 3.91060668},
          {1.82236170, 7.55361435, 4.25155932, 2.07941742},
          {5.67700371, 0.31313389, 8.42284790, 4.49754188},
  };

  eigen::matrix T {
          {0.00000000, 1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 0.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 293.43963623046875, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 4.163419723510742, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -5.645654201507568, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -5.645654678344727, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.011839533224701881, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
          {5.97900019, 9.21874243, 0.88492593, 1.95982943},
          {0.45227384, 3.25330398, 3.88677351, 2.71349105},
          {8.28737526, 3.56753391, 2.80934582, 5.42696129},
  };

  eigen::matrix T {
          {1.00000000, 0.00000000, 0.00000000, 0.00000000},
          {1.00000000, 0.00000000, 0.00000000, 0.00000000},
          {0.00000000, 0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 261.8872375488281, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 12.886770248413086, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -2.027737617492676, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -2.027737617492676, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.5533918142318726, Y, T);
}
