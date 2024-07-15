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
  scalar epsilon = std::is_same<scalar, double>::value ? scalar(0.0000001) : scalar(0.0001);
  CHECK(expected == doctest::Approx(L).epsilon(epsilon));
}

TEST_CASE("test_loss1")
{
  eigen::matrix Y {
          {5.870679683855396, 0.18078924941914432, 4.803890569757266, 8.361443615518427},
          {2.762725967294043, 5.145904161420163, 2.1377606268074216, 3.6716373675777563},
          {0.5230518225501912, 3.235352344784192, 4.41490399941436, 2.3646722099324333}
  };

  eigen::matrix T {
          {0, 0, 1, 0},
          {0, 1, 0, 0},
          {0, 1, 0, 0}
  };

  test_loss("cross_entropy_loss", cross_entropy_loss(), -4.381765365600586, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.05256272479891777, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -4.381765365600586, Y, T);
  test_loss("squared_error_loss", squared_error_loss(), 192.1064453125, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.534232139587402, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
          {2.042741429242198, 9.189160369204675, 0.022162317282340718, 6.926282528208948},
          {8.87855132535082, 8.75206559642744, 9.93563725104797, 1.61857948184018},
          {4.3337845750047945, 5.340551275551119, 8.146339686191746, 4.199086718345623}
  };

  eigen::matrix T {
          {0, 0, 0, 1},
          {1, 0, 0, 0},
          {0, 1, 0, 0}
  };

  test_loss("cross_entropy_loss", cross_entropy_loss(), -5.794290542602539, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.005902394652366638, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -5.794290542602539, Y, T);
  test_loss("squared_error_loss", squared_error_loss(), 485.35894775390625, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 6.825676918029785, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
          {6.679296141079335, 4.590231890781054, 4.983459115051482, 5.097595452716134},
          {2.327297494781975, 7.158940222550041, 3.2937708736156344, 4.949212266986153},
          {6.008809576838614, 3.633956743125431, 7.143338705121138, 3.8335523272473258}
  };

  eigen::matrix T {
          {0, 0, 1, 0},
          {0, 0, 1, 0},
          {1, 0, 0, 0}
  };

  test_loss("cross_entropy_loss", cross_entropy_loss(), -4.591383934020996, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.04572257027029991, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -4.591383934020996, Y, T);
  test_loss("squared_error_loss", squared_error_loss(), 297.9766845703125, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 7.56727409362793, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
          {0.4084068337260123, 0.7041627960649302, 9.262006149073942, 4.031747732216782},
          {5.274541216821977, 0.039707200513841606, 7.677634471792445, 0.8244127799172434},
          {9.606213844618562, 7.470868846908303, 5.223540178319292, 9.117584256740574}
  };

  eigen::matrix T {
          {0, 0, 0, 1},
          {0, 1, 0, 0},
          {1, 0, 0, 0}
  };

  test_loss("cross_entropy_loss", cross_entropy_loss(), -0.4303872585296631, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.6911457777023315, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -0.43038737773895264, Y, T);
  test_loss("squared_error_loss", squared_error_loss(), 424.3040771484375, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 13.518119812011719, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
          {3.31666810523401, 1.8076855989806435, 0.6220801709137708, 5.199977868471362},
          {5.39491734445041, 4.4544457488431775, 0.2310485503258388, 9.287036788404333},
          {6.823542987600912, 7.077766242560397, 0.6050048922051032, 4.144161802186545}
  };

  eigen::matrix T {
          {0, 0, 0, 1},
          {0, 1, 0, 0},
          {0, 1, 0, 0}
  };

  test_loss("cross_entropy_loss", cross_entropy_loss(), -5.099515438079834, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 0.017904402688145638, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), -5.099515438079834, Y, T);
  test_loss("squared_error_loss", squared_error_loss(), 260.6756591796875, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.644117832183838, Y, T);
}
