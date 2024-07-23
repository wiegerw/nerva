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

  eigen::matrix Y(N, K);
  eigen::matrix DY(N, K);

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
    {0.37454012, 0.95071429},
    {0.73199391, 0.59865850},
    {0.15601864, 0.15599452},
    {0.05808361, 0.86617613},
    {0.60111499, 0.70807260},
  };

  eigen::matrix T {
    {0.00000000, 1.00000000, 0.00000000},
    {1.00000000, 0.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
    {0.00000000, 1.00000000, 0.00000000},
  };

  eigen::matrix W1 {
    {0.16048044, 0.35825378},
    {0.50463152, -0.62237841},
    {0.16706774, -0.51876700},
    {0.13476609, 0.47099063},
    {0.02794417, -0.15997612},
    {-0.10188604, -0.22527617},
  };

  eigen::matrix b1 {
    {-0.57381582, 0.46958971, 0.31309026, 0.55319005, 0.38372764, -0.30049479},
  };

  eigen::matrix W2 {
    {0.04391845, 0.17592925, -0.07418138, 0.12133780, 0.34800497, 0.28079784},
    {-0.01118679, -0.34719944, -0.31079185, 0.24866264, 0.12015112, -0.19697127},
    {-0.17528901, -0.21305986, -0.34019256, 0.23698428, 0.10112186, -0.15516852},
    {0.24649617, -0.10293734, 0.26929134, -0.10611577, 0.32058507, -0.02204809},
  };

  eigen::matrix b2 {
    {0.20867662, -0.04562366, -0.06657781, 0.38756141},
  };

  eigen::matrix W3 {
    {-0.49054316, 0.18386941, 0.05368949, 0.25338203},
    {0.31388429, 0.09920563, 0.08516803, 0.27491242},
    {-0.14138609, -0.30682307, -0.16577356, -0.21237689},
  };

  eigen::matrix b3 {
    {0.40558881, -0.34701368, -0.36167952},
  };

  eigen::matrix Y1 {
    {0.34526360, -0.09748828, -0.58622235},
    {0.27290592, -0.10054651, -0.51603264},
    {0.30183780, -0.09065950, -0.51769274},
    {0.35957462, -0.09883735, -0.58745533},
    {0.29621056, -0.09922957, -0.54303473},
  };

  eigen::matrix DY1 {
    {0.09822052, -0.13691625, 0.03869573},
    {-0.10665898, 0.06425165, 0.04240733},
    {0.09451767, -0.13616578, 0.04164812},
    {0.09900116, -0.13740286, 0.03840170},
    {0.09499292, -0.13603333, 0.04104041},
  };

  eigen::matrix Y2 {
    {0.33888084, -0.08755577, -0.59130698},
    {0.26663059, -0.09071493, -0.52110910},
    {0.29554862, -0.08147207, -0.52184713},
    {0.35332042, -0.08907717, -0.59242415},
    {0.28990293, -0.08937314, -0.54811108},
  };

  eigen::matrix DY2 {
    {0.09768912, -0.13622549, 0.03853637},
    {-0.10716683, 0.06493966, 0.04222719},
    {0.09400785, -0.13551985, 0.04151201},
    {0.09847926, -0.13672766, 0.03824839},
    {0.09447664, -0.13534428, 0.04086764},
  };

  scalar lr = 0.01;
  std::vector<long> sizes = {2, 6, 4, 3};
  long N = 5;
  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr, sizes, N);
}


//--- end generated code ---//
