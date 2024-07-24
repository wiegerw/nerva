// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file dataset_wrapper_test.cpp
/// \brief Tests for dataset_wrapper.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <random>
#include "nerva/datasets/make_dataset.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/training.h"

using namespace nerva;

TEST_CASE("test_mlp")
{
  std::mt19937 rng{std::random_device{}()};
  datasets::dataset data = datasets::make_dataset("chessboard", 100, rng);
  datasets::dataset_view data1 = datasets::make_dataset_view(data);

  eigen::matrix W1 {
    {3, 4},
    {0, 6}
  };
  eigen::matrix b1 {{7}, {2}};

  eigen::matrix W2 {
    {1, 0},
    {2, 9}
  };
  eigen::matrix b2 {{1}, {4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 0}
  };
  eigen::matrix b3 {{3}, {2}};

  long batch_size = 5;

  multilayer_perceptron M;
  {
    auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
    M.layers.push_back(layer1);
    set_linear_layer_optimizer(*layer1, "GradientDescent");
    layer1->W = W1;
    layer1->b = b1;

    auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
    M.layers.push_back(layer2);
    set_linear_layer_optimizer(*layer2, "GradientDescent");
    layer2->W = W2;
    layer2->b = b2;

    auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, batch_size);
    M.layers.push_back(layer3);
    set_linear_layer_optimizer(*layer3, "GradientDescent");
    layer3->W = W3;
    layer3->b = b3;
  }

  auto accuracy1 = compute_accuracy(M, data.Xtrain, data.Ttrain, batch_size);
  auto accuracy2 = compute_accuracy(M, data1.Xtrain, data1.Ttrain, batch_size);

  CHECK(std::fabs(accuracy1 - accuracy2) < 0.00001);
}
