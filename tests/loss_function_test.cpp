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
    {0.36742274, 0.35949028, 0.27308698},
    {0.30354068, 0.41444678, 0.28201254},
    {0.34972793, 0.32481684, 0.32545523},
    {0.34815459, 0.44543710, 0.20640831},
    {0.19429503, 0.32073754, 0.48496742},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.2447052001953125, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.419629096984863, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.283669471740723, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.283669471740723, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.666532516479492, Y, T);
}

TEST_CASE("test_loss2")
{
  eigen::matrix Y {
    {0.52777285, 0.20090417, 0.27132298},
    {0.26273163, 0.32264205, 0.41462632},
    {0.45460889, 0.18870998, 0.35668113},
    {0.22136510, 0.49770545, 0.28092945},
    {0.44595740, 0.26068802, 0.29335457},
  };

  eigen::matrix T {
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.086756706237793, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.313835620880127, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.116687774658203, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.116687774658203, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.622492551803589, Y, T);
}

TEST_CASE("test_loss3")
{
  eigen::matrix Y {
    {0.28144839, 0.33493659, 0.38361502},
    {0.30910451, 0.20501579, 0.48587971},
    {0.43972324, 0.31403295, 0.24624382},
    {0.33462111, 0.36531756, 0.30006132},
    {0.26852208, 0.28904530, 0.44243262},
  };

  eigen::matrix T {
    {1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.5129189491271973, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.554657459259033, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.708432197570801, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.708432674407959, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.7234067916870117, Y, T);
}

TEST_CASE("test_loss4")
{
  eigen::matrix Y {
    {0.40741932, 0.26165398, 0.33092670},
    {0.39946334, 0.35773190, 0.24280476},
    {0.28962488, 0.31400862, 0.39636650},
    {0.44327877, 0.23078894, 0.32593229},
    {0.28449915, 0.26325951, 0.45224133},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.6121771335601807, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.6078267097473145, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.872381210327148, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.872381210327148, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.7460596561431885, Y, T);
}

TEST_CASE("test_loss5")
{
  eigen::matrix Y {
    {0.32584078, 0.39945709, 0.27470213},
    {0.34529719, 0.34289481, 0.31180800},
    {0.20006306, 0.40980735, 0.39012959},
    {0.29103148, 0.26184147, 0.44712705},
    {0.32969397, 0.43886940, 0.23143662},
  };

  eigen::matrix T {
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 0.00000000, 1.00000000},
    {0.00000000, 1.00000000, 0.00000000},
  };

  test_loss("squared_error_loss", squared_error_loss(), 3.2960329055786133, Y, T);
  test_loss("softmax_cross_entropy_loss", softmax_cross_entropy_loss(), 5.448777198791504, Y, T);
  test_loss("negative_log_likelihood_loss", negative_log_likelihood_loss(), 5.524302959442139, Y, T);
  test_loss("cross_entropy_loss", cross_entropy_loss(), 5.524302959442139, Y, T);
  test_loss("logistic_cross_entropy_loss", logistic_cross_entropy_loss(), 2.68286395072937, Y, T);
}


//--- end generated code ---//
