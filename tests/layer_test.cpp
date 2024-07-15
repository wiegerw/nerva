// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file layer_test.cpp
/// \brief Tests for layers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/random.h"
#include "nerva/neural_networks/weights.h"
#include <random>

using namespace nerva;

struct counter
{
  scalar i = 0;

  scalar operator()()
  {
    i += 1;
    return i;
  }
};

TEST_CASE("test_linear_layer1")
{
  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  eigen::matrix W {
    {3, 4},
    {5, 6}
  };

  eigen::vector b {{7, 2}};

  long D = 2;
  long N = 2;
  long K = 2;
  scalar density = 1;

  sparse_linear_layer layer(D, K, N);
  set_support_random(layer, density, rng);

  // N.B. THIS DOES NOT WORK
  // initialize_weights(weight_initialization::xavier, layer.W, layer.b, rng);

  std::cout << "========================" << std::endl;
  counter f;
  auto& W_values = const_cast<std::vector<scalar>&>(layer.W.values());
  for (auto& value: W_values)
  {
    value = f();
  }
  std::cout << "layer.W =\n" << layer.W.to_string() << std::endl;
  std::cout << "layer.W.values = " << print_list(layer.W.values()) << std::endl;
  std::vector<scalar> values = {1, 2, 3, 4};
  CHECK_EQ(values, layer.W.values());

  eigen::matrix W1_expected {
    {1, 2},
    {3, 4}
  };
  eigen::matrix W1 = mkl::to_eigen(layer.W);
  print_cpp_matrix("W1", W1);
  CHECK_EQ(W1_expected, W1);

  layer.W = mkl::to_csr<scalar>(W1_expected);
  std::cout << "layer.W =\n" << layer.W.to_string() << std::endl;
  std::cout << "layer.W.values = " << print_list(layer.W.values()) << std::endl;
  CHECK_EQ(values, layer.W.values());
}

TEST_CASE("test_linear_layer2")
{
  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  eigen::matrix W {
    {3, 4},
    {5, 6}
  };

  eigen::vector b {{7, 2}};

  long D = 2;
  long N = 2;
  long K = 2;
  scalar density = 1;

  sparse_linear_layer layer(D, K, N);
  set_support_random(layer, density, rng);
  initialize_weights(weight_initialization::xavier, layer.W, layer.b, rng);

  std::cout << "layer.W =\n" << layer.W.to_string() << std::endl;
  std::cout << "layer.W.values = " << print_list(layer.W.values()) << std::endl;
}

template <typename Layer1, typename Layer2>
void test_feedforward(Layer1& layer1, Layer2& layer2, const eigen::matrix& X)
{
  long N = X.rows();
  long K = layer1.output_size();

  print_cpp_matrix("X", X);

  // do a feedforward pass
  eigen::matrix Y1(N, K);
  layer1.X = X;
  layer1.feedforward(Y1);

  eigen::matrix Y2(N, K);
  layer2.X = X;
  layer2.feedforward(Y2);

  print_cpp_matrix("Y1", Y1);
  print_cpp_matrix("Y2", Y2);

  CHECK_LE((Y2 - Y1).squaredNorm(), 1e-10);
}

template <typename Layer1, typename Layer2>
void test_backpropagate(Layer1& layer1, Layer2& layer2, const eigen::matrix& Y, const eigen::matrix& DY)
{
  layer1.backpropagate(Y, DY);
  layer2.backpropagate(Y, DY);

  eigen::matrix W1 = mkl::to_eigen(layer1.W);
  eigen::matrix W2 = layer2.W;

  print_cpp_matrix("W1", W1);
  print_cpp_matrix("W2", W2);

  CHECK_LE((W2 - W1).squaredNorm(), 1e-10);
}

void test_layers(const eigen::matrix& W, const eigen::matrix& b, const eigen::matrix& X, const eigen::matrix& Y, const eigen::matrix& DY)
{
  std::cout << "=================" << std::endl;
  std::cout << "=== test_layers ===" << std::endl;
  std::cout << "=================" << std::endl;

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  long D = X.cols();
  long N = X.rows();
  long K = W.rows();

  sparse_linear_layer linear_layer1(D, K, N);
  linear_layer1.W = mkl::to_csr<scalar>(W);
  linear_layer1.DW = linear_layer1.W;
  linear_layer1.b = b;

  linear_layer<eigen::matrix> linear_layer2(D, K, N);
  linear_layer2.W = W;
  linear_layer2.b = b;

  test_feedforward(linear_layer1, linear_layer2, X);
  test_backpropagate(linear_layer1, linear_layer2, Y, DY);

  sparse_linear_layer linear_layer3(D, K, N);
  set_support_random(linear_layer3, scalar(1), rng);

  linear_layer<eigen::matrix> linear_layer4(D, K, N);
  linear_layer4.W = mkl::to_eigen<scalar>(linear_layer3.W);
  linear_layer4.b = linear_layer3.b;

  test_feedforward(linear_layer3, linear_layer4, X);
  test_backpropagate(linear_layer3, linear_layer4, Y, DY);

  relu_layer<mkl::sparse_matrix_csr<scalar>> relu_layer1(D, K, N);
  relu_layer1.W = mkl::to_csr<scalar>(W);
  relu_layer1.DW = relu_layer1.W;
  relu_layer1.b = b;

  relu_layer<eigen::matrix> relu_layer2(D, K, N);
  relu_layer2.W = W;
  relu_layer2.b = b;

  test_feedforward(relu_layer1, relu_layer2, X);
  test_backpropagate(relu_layer1, relu_layer2, Y, DY);

  sigmoid_layer<mkl::sparse_matrix_csr<scalar>> sigmoid_layer1(D, K, N);
  sigmoid_layer1.W = mkl::to_csr<scalar>(W);
  sigmoid_layer1.DW = sigmoid_layer1.W;
  sigmoid_layer1.b = b;

  sigmoid_layer<eigen::matrix> sigmoid_layer2(D, K, N);
  sigmoid_layer2.W = W;
  sigmoid_layer2.b = b;

  test_feedforward(sigmoid_layer1, sigmoid_layer2, X);
  test_backpropagate(sigmoid_layer1, sigmoid_layer2, Y, DY);

  softmax_layer<mkl::sparse_matrix_csr<scalar>> softmax_layer1(D, K, N);
  softmax_layer1.W = mkl::to_csr<scalar>(W);
  softmax_layer1.DW = softmax_layer1.W;
  softmax_layer1.b = b;

  softmax_layer<eigen::matrix> softmax_layer2(D, K, N);
  softmax_layer2.W = W;
  softmax_layer2.b = b;

  test_feedforward(softmax_layer1, softmax_layer2, X);
  test_backpropagate(softmax_layer1, softmax_layer2, Y, DY);
}

template <typename LossFunction>
void test_mlp(multilayer_perceptron& M1, multilayer_perceptron& M2, const eigen::matrix& X, const eigen::matrix& T, LossFunction loss)
{
  std::cout << "=================" << std::endl;
  std::cout << "=== test_mlp ===" << std::endl;
  std::cout << "=================" << std::endl;

  print_cpp_matrix("X", X);
  print_cpp_matrix("T", T);

//  M1.info("M1 before");
//  M2.info("M2 before");

  long K = T.cols();
  long N = X.rows();

  eigen::matrix Y1(N, K);
  eigen::matrix Y2(N, K);

  M1.layers.front()->X = X;
  M1.feedforward(Y1);
  eigen::matrix DY1 = loss.gradient(Y1, T);
  M1.backpropagate(Y1, DY1);

  M2.layers.front()->X = X;
  M2.feedforward(Y2);
  eigen::matrix DY2 = loss.gradient(Y2, T);
  M2.backpropagate(Y2, DY2);

  CHECK_LE((Y2 - Y1).squaredNorm(), 1e-10);
  CHECK_LE((DY2 - DY1).squaredNorm(), 1e-10);

  print_cpp_matrix("DY1", DY1);
  print_cpp_matrix("Y1", Y1);

//  M1.info("M1 after");
//  M2.info("M2 after");

  // optimize
  scalar eta = 0.01;
  M1.optimize(eta);
  M2.optimize(eta);

  // do another feedforward step
  M1.feedforward(Y1);
  M2.feedforward(Y2);

  CHECK_LE((Y2 - Y1).squaredNorm(), 1e-10);

  print_cpp_matrix("Y1", Y1);
}

inline
eigen::matrix random_matrix(long rows, long cols, scalar a = 1, scalar b = 5)
{
  eigen::matrix ones = eigen::ones<eigen::matrix>(rows, cols);
  scalar factor = (b - a) / scalar(2);

  eigen::matrix A = eigen::matrix::Random(rows, cols);   // Range [-1, 1]
  return factor * (A + ones) + a * ones;                 // Range [a, b]
}

inline
Eigen::VectorXi random_integer_vector(long size, long N, std::mt19937& gen)
{
  if (size <= 0 || N <= 0)
  {
    throw std::invalid_argument("Size and N must be positive integers.");
  }

  std::uniform_int_distribution<> dist(0, N-1);
  Eigen::VectorXi result(size);
  for (int i = 0; i < size; ++i)
  {
    result[i] = dist(gen);
  }
  return result;
}

// Creates a one hot encoding. Each column contains one value 1 and all other values 0.
inline
eigen::matrix random_target(long rows, long cols)
{
  long size = rows;
  long num_classes = cols;
  Eigen::VectorXi targets = random_integer_vector(size, num_classes, nerva_rng);
  return eigen::to_one_hot_rowwise(targets, num_classes);
}

void test_layers(long D, long N, long K)
{
  eigen::matrix X = random_matrix(N, D);
  eigen::matrix Y = random_matrix(N, K);
  eigen::matrix DY = random_matrix(N, K);
  eigen::matrix W = random_matrix(K, D);
  eigen::matrix b = random_matrix(1, K);
  test_layers(W, b, X, Y, DY);
}

TEST_CASE("test_layers")
{
  test_layers(3, 2, 2);
  test_layers(2, 3, 2);
  test_layers(2, 2, 3);
}

template <typename LossFunction>
void test_mlp(long D, long K1, long K2, long K3, long N, LossFunction loss)
{
  eigen::matrix X = random_matrix(N, D, 0.0, 1.0);  // the input of the MLP
  eigen::matrix T = random_target(N, K3);  // the target
  eigen::matrix W1 = random_matrix(K1, D, 0.0, 1.0);
  eigen::matrix W2 = random_matrix(K2, K1, 0.0, 1.0);
  eigen::matrix W3 = random_matrix(K3, K2, 0.0, 1.0);
  eigen::matrix b1 = eigen::matrix::Zero(1, K1);
  eigen::matrix b2 = eigen::matrix::Zero(1, K2);
  eigen::matrix b3 = eigen::matrix::Zero(1, K3);
  long batch_size = N;

  // Create dense MLP M1
  multilayer_perceptron M1;
  {
    auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(D, K1, batch_size);
    layer1->W = W1;
    layer1->b = b1;
    layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
    M1.layers.push_back(layer1);

    auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(K1, K2, batch_size);
    layer2->W = W2;
    layer2->b = b2;
    layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
    M1.layers.push_back(layer2);

    auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(K2, K3, batch_size);
    layer3->W = W3;
    layer3->b = b3;
    layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW,layer3->b, layer3->Db);
    M1.layers.push_back(layer3);
  }

  // Create sparse MLP M2
  multilayer_perceptron M2;
  {
    using matrix_t = mkl::sparse_matrix_csr<scalar>;
    auto layer1 = std::make_shared<relu_layer<matrix_t>>(D, K1, batch_size);
    layer1->W = mkl::to_csr<scalar>(W1);
    layer1->DW = layer1->W;
    layer1->b = b1;
    layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
    M2.layers.push_back(layer1);

    auto layer2 = std::make_shared<relu_layer<matrix_t>>(K1, K2, batch_size);
    layer2->W = mkl::to_csr<scalar>(W2);
    layer2->DW = layer2->W;
    layer2->b = b2;
    layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
    M2.layers.push_back(layer2);

    auto layer3 = std::make_shared<linear_layer<matrix_t>>(K2, K3, batch_size);
    layer3->W = mkl::to_csr<scalar>(W3);
    layer3->DW = layer3->W;
    layer3->b = b3;
    layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
    M2.layers.push_back(layer3);
  }

  test_mlp(M1, M2, X, T, loss);
}

TEST_CASE("test_mlp")
{
  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(4, 2, 3, 2, 5, loss1);
  test_mlp(4, 2, 3, 2, 5, loss2);
  test_mlp(4, 2, 3, 2, 5, loss3);
  test_mlp(4, 2, 3, 2, 5, loss4);
  test_mlp(6, 5, 7, 3, 10, loss1);
  test_mlp(6, 5, 7, 3, 10, loss2);
  test_mlp(6, 5, 7, 3, 10, loss3);
  test_mlp(6, 5, 7, 3, 10, loss4);
}

/*
TEST_CASE("test_mlp1")
{
  eigen::matrix X {
    {1, 2},
    {3, 4},
    {5, 6},
    {7, 8}
  };

  eigen::matrix T {
    {1, 0},
    {1, 0},
    {0, 1},
    {1, 0}
  };

  eigen::matrix W1 {
    {3, 4},
    {5, 6}
  };
  eigen::matrix b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 1},
    {2, 9}
  };
  eigen::matrix b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 4}
  };
  eigen::matrix b3 {{3, 2}};

  long batch_size = X.cols();

  // Create dense MLP M1
  multilayer_perceptron M1;
  {
    auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
    M1.layers.push_back(layer1);
    layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
    layer1->W = W1;
    layer1->b = b1;

    auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
    M1.layers.push_back(layer2);
    layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
    layer2->W = W2;
    layer2->b = b2;

    auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, batch_size);
    M1.layers.push_back(layer3);
    layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
    layer3->W = W3;
    layer3->b = b3;
  }

  // Create sparse MLP M2
  multilayer_perceptron M2;
  {
    using matrix_t = mkl::sparse_matrix_csr<scalar>;
    auto layer1 = std::make_shared<relu_layer<matrix_t>>(2, 2, batch_size);
    M2.layers.push_back(layer1);
    layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
    layer1->W = mkl::to_csr<scalar>(W1);
    layer1->DW = layer1->W;
    layer1->b = b1;

    auto layer2 = std::make_shared<relu_layer<matrix_t>>(2, 2, batch_size);
    M2.layers.push_back(layer2);
    layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
    layer2->W = mkl::to_csr<scalar>(W2);
    layer2->DW = layer2->W;
    layer2->b = b2;

    auto layer3 = std::make_shared<linear_layer<matrix_t>>(2, 2, batch_size);
    M2.layers.push_back(layer3);
    layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
    layer3->W = mkl::to_csr<scalar>(W3);
    layer3->DW = layer3->W;
    layer3->b = b3;
  }

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M1, M2, X, T, loss1);
  test_mlp(M1, M2, X, T, loss2);
  test_mlp(M1, M2, X, T, loss3);
  // test_mlp(M1, X, T, loss4); // TODO: this leads to numerical problems: log(0)
}

TEST_CASE("test_mlp2")
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
    {0, 6}
  };
  eigen::vector b1 {{7, 2}};

  eigen::matrix W2 {
    {1, 0},
    {2, 9}
  };
  eigen::vector b2 {{1, 4}};

  eigen::matrix W3 {
    {4, 1},
    {2, 0}
  };
  eigen::vector b3 {{3, 2}};

  long batch_size = X.cols();

  // Create dense MLP M1
  multilayer_perceptron M1;
  {
    auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
    M1.layers.push_back(layer1);
    layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
    layer1->W = W1;
    layer1->b = b1;

    auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(2, 2, batch_size);
    M1.layers.push_back(layer2);
    layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
    layer2->W = W2;
    layer2->b = b2;

    auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(2, 2, batch_size);
    M1.layers.push_back(layer3);
    layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
    layer3->W = W3;
    layer3->b = b3;
  }

  // Create sparse MLP M2
  multilayer_perceptron M2;
  {
    using matrix_t = mkl::sparse_matrix_csr<scalar>;
    auto layer1 = std::make_shared<relu_layer<matrix_t>>(2, 2, batch_size);
    M2.layers.push_back(layer1);
    layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
    layer1->W = mkl::to_csr<scalar>(W1);
    layer1->DW = layer1->W;
    layer1->b = b1;

    auto layer2 = std::make_shared<relu_layer<matrix_t>>(2, 2, batch_size);
    M2.layers.push_back(layer2);
    layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
    layer2->W = mkl::to_csr<scalar>(W2);
    layer2->DW = layer2->W;
    layer2->b = b2;

    auto layer3 = std::make_shared<linear_layer<matrix_t>>(2, 2, batch_size);
    M2.layers.push_back(layer3);
    layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<matrix_t>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
    layer3->W = mkl::to_csr<scalar>(W3);
    layer3->DW = layer3->W;
    layer3->b = b3;
  }

  squared_error_loss loss1;
  cross_entropy_loss loss2;
  logistic_cross_entropy_loss loss3;
  softmax_cross_entropy_loss loss4;

  test_mlp(M1, M2, X, T, loss1);
  test_mlp(M1, M2, X, T, loss2);
  test_mlp(M1, M2, X, T, loss3);
  // test_mlp(M1, X, T, loss4); // TODO: this leads to numerical problems: log(0)
}
*/