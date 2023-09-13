// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file gradient_test.cpp
/// \brief Tests for gradients.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/batch_normalization_layers_colwise.h"
#include "nerva/neural_networks/check_gradients.h"
#include "nerva/datasets/make_dataset.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/loss_functions_colwise.h"
#include "nerva/neural_networks/mlp_algorithms.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include <random>
#include <type_traits>

using namespace nerva;

template <typename Layer, typename LossFunction>
void test_linear_layer(Layer& layer, const eigen::matrix& X, const eigen::matrix& T, LossFunction loss)
{
  long K = T.rows();
  long N = X.rows();

  // do a feedforward + backpropagate pass to compute Db and DW
  layer.X = X;
  eigen::matrix Y(K, N);
  layer.feedforward(Y);
  eigen::matrix dY = loss.gradient(Y, T);
  layer.backpropagate(Y, dY);

  auto f = [&]()
  {
    layer.feedforward(Y);
    return loss(Y, T);
  };

  // check the gradients of b and W
  scalar h = std::is_same<scalar, double>::value ? scalar(0.0001) : scalar(0.01);
  CHECK(check_gradient("Db", f, layer.b, layer.Db, h));
  CHECK(check_gradient("DW", f, layer.W, layer.DW, h));
}

// test a layer with parameters beta and gamma
template <typename Layer, typename LossFunction>
void test_affine_layer(Layer& layer, const eigen::matrix& X, const eigen::matrix& T, LossFunction loss)
{
  long K = T.rows();
  long N = X.rows();

  // do a feedforward + backpropagate pass to compute Db and DW
  layer.X = X;
  eigen::matrix Y(K, N);
  layer.feedforward(Y);
  eigen::matrix dY = loss.gradient(Y, T);
  layer.backpropagate(Y, dY);

  auto f = [&]()
  {
    layer.feedforward(Y);
    return loss(Y, T);
  };

  scalar h = std::is_same<scalar, double>::value ? scalar(0.0001) : scalar(0.01);
  CHECK(check_gradient("Dbeta", f, layer.beta, layer.Dbeta, h));
  CHECK(check_gradient("Dgamma", f, layer.gamma, layer.Dgamma, h));
}

template <typename LossFunction>
void test_mlp(multilayer_perceptron& M, const eigen::matrix& X, const eigen::matrix& T, LossFunction loss)
{
  std::cout << "=================" << std::endl;
  std::cout << "=== test_mlp ===" << std::endl;
  std::cout << "=================" << std::endl;

  long K = T.rows();
  long N = X.cols();
  eigen::matrix Y(K, N);

  // do a feedforward + backpropagate pass to compute Db and DW
  M.layers.front()->X = X;
  M.feedforward(Y);
  eigen::matrix dY = loss.gradient(Y, T);
  M.backpropagate(Y, dY);

  if (false)
  {
    print_cpp_matrix("X", X);
    print_cpp_matrix("T", T);
    print_cpp_matrix("Y", Y);
    print_cpp_matrix("dY", dY);
  }

  auto f = [&]()
  {
    M.feedforward(Y);
    return loss(Y, T);
  };

  // check the gradients of b and W
  scalar h = std::is_same<scalar, double>::value ? scalar(0.0001) : scalar(0.01);

  unsigned int index = 1;
  for (const auto& layer: M.layers)
  {
    auto llayer = dynamic_cast<dense_linear_layer*>(layer.get());
    if (llayer)
    {
      std::string i = std::to_string(index++);
      CHECK(check_gradient("DW" + i, f, llayer->W, llayer->DW, h));
      CHECK(check_gradient("Db" + i, f, llayer->b, llayer->Db, h));
    }

    auto blayer = dynamic_cast<dense_batch_normalization_layer*>(layer.get());
    if (blayer)
    {
      CHECK(check_gradient("Dbeta", f, blayer->beta, blayer->Dbeta, h));
      CHECK(check_gradient("Dgamma", f, blayer->gamma, blayer->Dgamma, h));
    }
  }
}

TEST_CASE("test_linear_layer1")
{
  eigen::matrix X {
    {1, 2, 7},
    {3, 4, 5}
  };

  eigen::matrix T {
    {1, 1, 0},
    {0, 0, 1}
  };

  eigen::matrix W {
    {3, 4},
    {5, 6}
  };

  eigen::vector b {{7, 2}};

  long D = X.rows();
  long N = X.cols();
  long K = W.cols();

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  linear_layer<eigen::matrix> layer1(D, K, N);
  layer1.W = W;
  layer1.b = b;
  test_linear_layer(layer1, X, T, loss1);
  test_linear_layer(layer1, X, T, loss2);
  test_linear_layer(layer1, X, T, loss3);
  test_linear_layer(layer1, X, T, loss4);

  relu_layer<eigen::matrix> layer2(D, K, N);
  layer2.W = W;
  layer2.b = b;
  test_linear_layer(layer2, X, T, loss1);
  test_linear_layer(layer2, X, T, loss2);
  test_linear_layer(layer2, X, T, loss3);
  test_linear_layer(layer2, X, T, loss4);

  sigmoid_layer<eigen::matrix> layer3(D, K, N);
  layer3.W = W;
  layer3.b = b;
  test_linear_layer(layer3, X, T, loss1);
  test_linear_layer(layer3, X, T, loss2);
  test_linear_layer(layer3, X, T, loss3);
  test_linear_layer(layer3, X, T, loss4);

  softmax_layer<eigen::matrix> layer4(D, K, N);
  layer4.W = W;
  layer4.b = b;
  test_linear_layer(layer4, X, T, loss1);
  test_linear_layer(layer4, X, T, loss2);
  test_linear_layer(layer4, X, T, loss3);
  test_linear_layer(layer4, X, T, loss4);
}

TEST_CASE("test_linear_layer2")
{
  eigen::matrix X {
    {-0.602793, 1,     -1    },
    {  -1,      1,  0.0728062}
  };

  eigen::matrix T {
    {0, 1, 1},
    {1, 0, 0}
  };

  eigen::matrix W {
    {0.047371,  0.465252},
    {-0.511825,  0.401837}
  };

  eigen::vector b {{0, 0}};

  long D = X.rows();
  long N = X.cols();
  long K = W.cols();

  squared_error_loss loss1;

  relu_layer<eigen::matrix> layer1(D, K, N);
  layer1.W = W;
  layer1.b = b;
  test_linear_layer(layer1, X, T, loss1);
}

TEST_CASE("test_dropout_layer1")
{
  eigen::matrix X {
    {1, 2, 7},
    {3, 4, 5}
  };

  eigen::matrix T {
    {1, 1, 0},
    {0, 0, 1}
  };

  eigen::matrix W {
    {3, 4},
    {5, 6}
  };

  eigen::vector b {{7, 2}};

  long D = X.rows();
  long N = X.cols();
  long K = W.cols();

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  scalar p = 0.8;

  linear_dropout_layer<eigen::matrix> layer1(D, K, N, p);
  layer1.W = W;
  layer1.b = b;
  test_linear_layer(layer1, X, T, loss1);
  // test_linear_layer(layer1, X, T, loss2);  // TODO: fails with NaN
  test_linear_layer(layer1, X, T, loss3);
  test_linear_layer(layer1, X, T, loss4);

  relu_dropout_layer<eigen::matrix> layer2(D, K, N, p);
  layer2.W = W;
  layer2.b = b;
  test_linear_layer(layer2, X, T, loss1); // DW[1,1] =    3.95816e-10    4.26326e-10              0              0              0              0
  test_linear_layer(layer2, X, T, loss2); // DW[1,1] =   -5.20811e-11   -4.44089e-11              0              0              0              0
  test_linear_layer(layer2, X, T, loss3);
  test_linear_layer(layer2, X, T, loss4);

  sigmoid_dropout_layer<eigen::matrix> layer3(D, K, N, p);
  layer3.W = W;
  layer3.b = b;
  test_linear_layer(layer3, X, T, loss1);
  test_linear_layer(layer3, X, T, loss2);
  test_linear_layer(layer3, X, T, loss3);
  test_linear_layer(layer3, X, T, loss4);
}

TEST_CASE("test_batch_normalization_layer1")
{
  eigen::matrix X {
    {1, 2, 7},
    {3, 4, 5}
  };

  eigen::matrix T {
    {1, 1, 0},
    {0, 0, 1}
  };

  eigen::matrix W {
    {3, 4},
    {5, 6}
  };

  eigen::vector b {{7, 2}};

  long D = X.rows();
  long N = X.cols();

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  batch_normalization_layer layer1(D, N);
  test_affine_layer(layer1, X, T, loss1);
  // test_affine_layer(layer1, X, T, loss2); // TODO: this fails with NaN
  test_affine_layer(layer1, X, T, loss3);
  test_affine_layer(layer1, X, T, loss4);
}

TEST_CASE("test_affine_layer1")
{
  eigen::matrix X {
    {1, 2, 7},
    {3, 4, 5}
  };

  eigen::matrix T {
    {1, 1, 0},
    {0, 0, 1}
  };

  eigen::matrix beta {
    {3},
    {1}
  };

  eigen::matrix gamma {
    {2},
    {5}
  };

  long D = X.rows();
  long N = X.cols();

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  affine_layer layer1(D, N);
  layer1.beta = beta;
  layer1.gamma = gamma;

  test_affine_layer(layer1, X, T, loss1);
  test_affine_layer(layer1, X, T, loss2);
  test_affine_layer(layer1, X, T, loss3);
  test_affine_layer(layer1, X, T, loss4);
}

TEST_CASE("test_mlp1")
{
  eigen::matrix X {
    {1, 2, 7, 8},
    {3, 4, 5, 2}
  };

  eigen::matrix T {
    {1, 1, 0, 1},
    {0, 0, 1, 0}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::vector b3 {{3, 2}};

  long batch_size = 2;
  multilayer_perceptron M;

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, batch_size);
  M.layers.push_back(layer3);
  layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
  layer3->W = W3;
  layer3->b = b3;

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_linear_layer(*layer1, X, T, loss1);
  test_linear_layer(*layer1, X, T, loss2);
  test_linear_layer(*layer1, X, T, loss3);
  test_linear_layer(*layer1, X, T, loss4);

  test_mlp(M, X, T, loss1);
  test_mlp(M, X, T, loss2);
  test_mlp(M, X, T, loss3);
  // test_mlp(M, X, T, loss4); // TODO: this leads to numerical problems: log(0)
}

TEST_CASE("test_dropout_relu")
{
  std::mt19937 rng{78147218};
  scalar p = 0.8;

  eigen::matrix X {
    {1, 2, 5},
    {3, 4, 1}
  };

  eigen::matrix T {
    {1, 0, 0},
    {0, 1, 1}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::vector b3 {{3, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  auto layer1 = std::make_shared<relu_dropout_layer<eigen::matrix>>(2, 2, N, p);
  M.layers.push_back(layer1);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer3);
  layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
  layer3->W = W3;
  layer3->b = b3;

  renew_dropout_masks(M, rng);
  auto dlayer = dynamic_cast<relu_dropout_layer<eigen::matrix>*>(M.layers[0].get());
  if (dlayer)
  {
    std::cout << "relu dropout layer" << std::endl;
    std::cout << dlayer->R << std::endl;
  }

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  test_mlp(M, X, T, loss2);
  test_mlp(M, X, T, loss3);
  // test_mlp(M, X, T, loss4); // TODO: fails due to overflow
}

TEST_CASE("test_dropout_linear")
{
  std::mt19937 rng{78147218};
  scalar p = 0.8;

  eigen::matrix X {
    {1, 2, 5},
    {3, 4, 1}
  };

  eigen::matrix T {
    {1, 0, 0},
    {0, 1, 1}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::vector b3 {{3, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_dropout_layer<eigen::matrix>>(2, 2, N, p);
  M.layers.push_back(layer3);
  layer3->W = W3;
  layer3->b = b3;

  renew_dropout_masks(M, rng);
  auto dlayer = dynamic_cast<linear_dropout_layer<eigen::matrix>*>(M.layers[2].get());
  if (dlayer)
  {
    std::cout << "linear dropout layer" << std::endl;
    std::cout << dlayer->R << std::endl;
  }

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  test_mlp(M, X, T, loss2);
  test_mlp(M, X, T, loss3);
  // test_mlp(M, X, T, loss4); // TODO: fails due to overflow
}

TEST_CASE("test_dropout_sigmoid")
{
  std::mt19937 rng{78147218};
  scalar p = 0.8;

  eigen::matrix X {
    {1, 2, 5},
    {3, 4, 1}
  };

  eigen::matrix T {
    {1, 0, 0},
    {0, 1, 1}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::vector b3 {{3, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  auto layer1 = std::make_shared<sigmoid_dropout_layer<eigen::matrix>>(2, 2, N, p);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<sigmoid_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer3);
  layer3->W = W3;
  layer3->b = b3;

  renew_dropout_masks(M, rng);
  auto dlayer = dynamic_cast<sigmoid_dropout_layer<eigen::matrix>*>(M.layers[0].get());
  if (dlayer)
  {
    std::cout << "sigmoid dropout layer" << std::endl;
    std::cout << dlayer->R << std::endl;
  }

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  test_mlp(M, X, T, loss2);
  test_mlp(M, X, T, loss3);
  // test_mlp(M, X, T, loss4); // TODO: fails due to overflow
}

TEST_CASE("test_batch_normalization1")
{
  eigen::matrix X {
    {1, 2, 7, 8},
    {3, 4, 5, 2}
  };

  eigen::matrix T {
    {1, 1, 0, 1},
    {0, 0, 1, 0}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::vector b3 {{3, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  M.layers.push_back(std::make_shared<batch_normalization_layer>(2, N));

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = W2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer3);
  layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
  layer3->W = W3;
  layer3->b = b3;

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  test_mlp(M, X, T, loss2);
  test_mlp(M, X, T, loss3);
  test_mlp(M, X, T, loss4);
}

TEST_CASE("test_batch_normalization2")
{
  eigen::matrix X {
    {1, 2},
    {3, 4}
  };

  eigen::matrix T {
    {1, 0},
    {0, 1}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::vector b3 {{3, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  M.layers.push_back(std::make_shared<batch_normalization_layer>(2, N));

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
//  test_mlp(M, X, T, loss2);
  test_mlp(M, X, T, loss3);
  test_mlp(M, X, T, loss4);
}

TEST_CASE("test_simple_batch_normalization1")
{
  eigen::matrix X {
    {1, 2, 7, 8},
    {3, 4, 5, 2}
  };

  eigen::matrix T {
    {1, 1, 0, 1},
    {0, 0, 1, 0}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  auto layer1 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  M.layers.push_back(std::make_shared<simple_batch_normalization_layer>(2, N));

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  // test_mlp(M, X, T, loss2); // TODO: this fails with NaN!
  test_mlp(M, X, T, loss3);
  test_mlp(M, X, T, loss4);
}

TEST_CASE("test_simple_batch_normalization2")
{
  eigen::matrix X {
    {1, 2, 7, 8},
    {3, 4, 5, 2}
  };

  eigen::matrix T {
    {1, 1, 0, 1},
    {0, 0, 1, 0}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::vector b1 {{7, 2}};

  long N = X.cols();
  multilayer_perceptron M;

  M.layers.push_back(std::make_shared<simple_batch_normalization_layer>(2, N));

  auto layer1 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, N);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = W1;
  layer1->b = b1;

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  // test_mlp(M, X, T, loss2); // TODO: this fails with NaN!
  test_mlp(M, X, T, loss3);
  test_mlp(M, X, T, loss4);
}

TEST_CASE("test_chessboard")
{
  log::logger::set_reporting_level(log::debug);

  long batch_size = 2;
  long N = 3;
  std::mt19937 rng{1885661379};
  auto [X, T] = datasets::make_dataset_chessboard(N, rng);

  multilayer_perceptron M;

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  initialize_weights(weight_initialization::xavier, layer1->W, layer1->b, rng);

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  initialize_weights(weight_initialization::xavier, layer2->W, layer2->b, rng);

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, batch_size);
  M.layers.push_back(layer3);
  layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
  initialize_weights(weight_initialization::xavier, layer3->W, layer3->b, rng);

  squared_error_loss loss1;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M, X, T, loss1);
  test_mlp(M, X, T, loss3);
  test_mlp(M, X, T, loss4);

  // N.B. Do not use cross_entropy_loss, since it doesn't work with this architecture.
}

TEST_CASE("test_derivatives2")
{
  eigen::matrix X {
    {1, 2, 3},
    {3, 4, 1}
  };

  eigen::matrix Y {
    {1, 2},
    {3, 4}
  };

  eigen::matrix R
  {
    {5, 2},
    {3, 7}
  };

  auto fY = [](const eigen::matrix& Y)
  {
    return Y.squaredNorm() / scalar(2);
  };

  auto dfY = [](const eigen::matrix& Y)
  {
    return Y;
  };

  auto fR = [fY](const eigen::matrix& R)
  {
    long N = R.cols();
    eigen::vector Sigma = R.array().square().rowwise().sum() / N;
    eigen::matrix Y = Diag(eigen::power_minus_half(Sigma)) * R;
    return fY(Y);
  };

  auto dfR = [dfY](const eigen::matrix& R)
  {
    using eigen::diag;
    using eigen::Diag;

    long N = R.cols();
    eigen::vector Sigma = R.array().square().rowwise().sum() / N;
    eigen::matrix Y = Diag(eigen::power_minus_half(Sigma)) * R;
    eigen::matrix DY = dfY(Y);
    eigen::matrix DiagDYY = Diag(diag(DY * Y.transpose())) * Y;
    eigen::matrix DR = Diag(eigen::power_minus_half(Sigma)) * (-DiagDYY / N + DY);
    return DR;
  };

  scalar h = std::is_same<scalar, double>::value ? scalar(0.0001) : scalar(0.01);

  eigen::matrix DY1 = approximate_derivative([&]() { return fY(Y); }, Y, h);
  eigen::matrix DY2 = dfY(Y);
  print_cpp_matrix("DY1", DY1);
  print_cpp_matrix("DY2", DY2);
  CHECK_LE((DY1 - DY2).squaredNorm(), 1e-8);

  eigen::matrix DR1 = approximate_derivative([&]() { return fR(R); }, R, h);
  eigen::matrix DR2 = dfR(R);
  print_cpp_matrix("DR1", DR1);
  print_cpp_matrix("DR2", DR2);
  CHECK_LE((DR1 - DR2).squaredNorm(), 1e-8);
}

TEST_CASE("test_power_minus_half")
{
  using eigen::Diag;

  eigen::matrix X {
    {6, 8, 10},
    {3, 12, 15},
  };

  eigen::vector Sigma {
    {4},
    {9}
  };

  eigen::matrix expected {
    {3, 4, 5},
    {1, 4, 5},
  };

  eigen::matrix Y = Diag(eigen::power_minus_half(Sigma)) * X;
  std::cout << "Y=\n" << Y << std::endl;

  CHECK_LE((expected - Y).squaredNorm(), 1e-10);
}
