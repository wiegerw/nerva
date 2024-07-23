// Copyright: Wieger Wesselink 2022 - 2024
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file multilayer_perceptron_test.cpp
/// \brief Tests for multilayer perceptrons.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/utilities/string_utility.h"
#include <iostream>

using namespace nerva;

inline
void check_equal_matrices(const std::string& name1, const eigen::matrix& X1, const std::string& name2, const eigen::matrix& X2, scalar epsilon = 1e-5)
{
  scalar error = (X2 - X1).squaredNorm();
  if (error > epsilon)
  {
    CHECK_LE(error, epsilon);
    print_cpp_matrix(name1, X1);
    print_cpp_matrix(name2, X2);
  }
}

void construct_mlp(multilayer_perceptron& M,
                   const eigen::matrix& W1,
                   const eigen::matrix& b1,
                   const eigen::matrix& W2,
                   const eigen::matrix& b2,
                   const eigen::matrix& W3,
                   const eigen::matrix& b3,
                   const std::vector<long>& sizes,
                   long N
                  )
{
  long batch_size = N;

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(sizes[0], sizes[1], batch_size);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(sizes[1], sizes[2], batch_size);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(sizes[2], sizes[3], batch_size);
  M.layers.push_back(layer3);
  layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
  layer3->W = W3;
  layer3->b = b3;
}

void test_mlp_execution(const eigen::matrix& X,
                        const eigen::matrix& T,
                        const eigen::matrix& W1,
                        const eigen::matrix& b1,
                        const eigen::matrix& W2,
                        const eigen::matrix& b2,
                        const eigen::matrix& W3,
                        const eigen::matrix& b3,
                        const eigen::matrix& Y1,
                        const eigen::matrix& DY1,
                        const eigen::matrix& Y2,
                        const eigen::matrix& DY2,
                        scalar lr,
                        const std::vector<long>& sizes,
                        long N
                       )
{
  multilayer_perceptron M;
  long K = sizes.back(); // the output size of the MLP
  construct_mlp(M, W1, b1, W2, b2, W3, b3, sizes, N);

  eigen::matrix Y(K, N);
  eigen::matrix DY(K, N);

  softmax_cross_entropy_loss loss;

  M.feedforward(X, Y);
  DY = loss.gradient(Y, T) / N; // take the average of the gradients in the batch

  check_equal_matrices("Y", Y, "Y1", Y1);
  check_equal_matrices("DY", DY, "DY1", DY1);

  M.backpropagate(Y, DY);
  M.optimize(lr);
  M.feedforward(X, Y);
  M.backpropagate(Y, DY);

  check_equal_matrices("Y", Y, "Y2", Y2);
  check_equal_matrices("DY", DY, "DY2", DY2);
}

//--- begin generated code ---//
TEST_CASE("test_mlp1")
{
  eigen::matrix X {
    {0.37454012, 0.73199391, 0.15601864, 0.05808361, 0.60111499},
    {0.95071429, 0.59865850, 0.15599452, 0.86617613, 0.70807260},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000},
    {1.00000000, 0.00000000, 1.00000000, 1.00000000, 1.00000000},
    {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000},
  };

  eigen::matrix W1 {
    {0.56910557, 0.07166054},
    {0.33410794, 0.50955588},
    {-0.38063282, -0.67608416},
    {0.10324049, 0.38635021},
    {0.03933064, -0.36663115},
    {-0.51061338, -0.20930530},
  };

  eigen::matrix b1 {
    {-0.01569078},
    {-0.67308903},
    {-0.28744614},
    {0.34278899},
    {0.42325342},
    {0.28628695},
  };

  eigen::matrix W2 {
    {-0.21767755, -0.03741430, -0.01714270, 0.17436612, -0.14338133, -0.08469867},
    {-0.00848581, 0.22186838, 0.16416024, 0.04421955, 0.08398556, -0.24910595},
    {0.10349254, -0.19608261, -0.35598740, -0.01798805, 0.29870310, 0.05946133},
    {0.29850718, 0.37467718, -0.16919829, 0.36069894, -0.10661299, -0.14757198},
  };

  eigen::matrix b2 {
    {0.27065653},
    {0.13387823},
    {-0.13294645},
    {0.10113561},
  };

  eigen::matrix W3 {
    {-0.21194729, 0.21646115, 0.38749391, 0.46850896},
    {-0.48358196, 0.33247125, -0.11519450, -0.14635921},
    {0.20089458, 0.13789035, 0.25491142, -0.39599109},
  };

  eigen::matrix b3 {
    {0.19588137},
    {0.43950686},
    {0.30333257},
  };

  eigen::matrix Y1 {
    {0.37142459, 0.38985717, 0.27270225, 0.32006717, 0.38177440},
    {0.25502959, 0.29406115, 0.31485462, 0.25317207, 0.28144985},
    {0.22832255, 0.21284100, 0.29881018, 0.26601246, 0.21966586},
  };

  eigen::matrix DY1 {
    {0.07254817, -0.12717782, 0.06515709, 0.06938004, 0.07259811},
    {-0.13542317, 0.06616984, -0.13203768, -0.13510932, -0.13433184},
    {0.06287500, 0.06100797, 0.06688060, 0.06572928, 0.06173372},
  };

  eigen::matrix Y2 {
    {0.37054998, 0.38905841, 0.27182075, 0.31913325, 0.38094819},
    {0.26373869, 0.30263132, 0.32268751, 0.26162237, 0.29005390},
    {0.22358359, 0.20813982, 0.29452613, 0.26143086, 0.21495876},
  };

  eigen::matrix DY2 {
    {0.07241082, -0.12731782, 0.06503753, 0.06925106, 0.07245933},
    {-0.13492474, 0.06666428, -0.13156864, -0.13461928, -0.13383637},
    {0.06251393, 0.06065353, 0.06653112, 0.06536821, 0.06137705},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {2, 6, 4, 3};
  long N = 5;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}


//--- end generated code ---//
