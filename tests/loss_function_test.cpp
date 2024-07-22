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
          {3.74540181, 9.50714311, 7.31993969},
          {5.98658524, 1.56018725, 1.55994605},
          {0.58083706, 8.66176159, 6.01115052},
          {7.08072607, 0.20584592, 9.69909855},
  };

  eigen::matrix T {
          {0.00000000, 1.00000000, 0.00000000},
          {0.00000000, 1.00000000, 0.00000000},
          {0.00000000, 1.00000000, 0.00000000},
          {1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 404.83148193359375, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 7.316563606262207, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -6.813143730163574, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -6.813143730163574, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.1917884796857834, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
          {6.17481548, 6.11653199, 0.07066405},
          {0.23062523, 5.24774708, 3.99861032},
          {0.46665759, 9.73755521, 2.32771417},
          {0.90606525, 6.18386047, 3.82462053},
  };

  eigen::matrix T {
          {1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 0.00000000, 1.00000000},
          {1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 0.00000000, 1.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 248.34161376953125, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 13.897750854492188, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -3.7857255935668945, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -3.7857253551483154, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.5286419987678528, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
          {6.80307571, 4.50499307, 0.13265060},
          {9.42201761, 5.63288262, 3.85416564},
          {0.15966351, 2.30893903, 2.41025542},
          {6.83263550, 6.09996697, 8.33194928},
  };

  eigen::matrix T {
          {0.00000000, 0.00000000, 1.00000000},
          {0.00000000, 1.00000000, 0.00000000},
          {1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 346.3843688964844, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 16.048355102539062, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 0.31781864166259766, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 0.3178187608718872, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 1.2513306140899658, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
          {6.62522318, 3.11711145, 5.20068069},
          {5.46710325, 1.84854537, 9.69584631},
          {7.75132846, 9.39498948, 8.94827361},
          {5.97900019, 9.21874243, 0.88492593},
  };

  eigen::matrix T {
          {0.00000000, 0.00000000, 1.00000000},
          {1.00000000, 0.00000000, 0.00000000},
          {0.00000000, 1.00000000, 0.00000000},
          {1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 509.8265380859375, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 9.79195499420166, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -7.3759684562683105, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -7.375967979431152, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.01232351828366518, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
          {3.88677351, 2.71349105, 8.28737526},
          {3.56753391, 2.80934582, 5.42696129},
          {1.40924311, 8.02197001, 0.74550736},
          {9.86886938, 7.72244792, 1.98715762},
  };

  eigen::matrix T {
          {0.00000000, 0.00000000, 1.00000000},
          {0.00000000, 0.00000000, 1.00000000},
          {0.00000000, 0.00000000, 1.00000000},
          {1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 324.43817138671875, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 7.6113176345825195, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -5.801807880401611, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), -5.801807880401611, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.39300477504730225, Y, T);
}
