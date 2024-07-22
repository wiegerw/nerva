// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file gradient_test.cpp
/// \brief Tests for gradients.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

// Gradient tests are done with double precision
#define NERVA_USE_DOUBLE

#include "doctest/doctest.h"
#include "nerva/neural_networks/batch_normalization_layers.h"
#include "nerva/neural_networks/check_gradients.h"
#include "nerva/datasets/make_dataset.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/mlp_algorithms.h"
#include "nerva/neural_networks/parse_layer.h"
#include "nerva/neural_networks/random.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include <random>
#include <type_traits>

using namespace nerva;

inline
void print_mlp(const std::string& name, const multilayer_perceptron& M)
{
  std::cout << "==================================\n";
  std::cout << "   " << name << "\n";
  std::cout << "==================================\n";
  for (const auto& layer: M.layers)
  {
    std::cout << layer->to_string() << std::endl;
  }
  std::cout << std::endl;
}

template <typename Layer, typename LossFunction>
void test_linear_layer(Layer& layer, const eigen::matrix& X, const eigen::matrix& T, LossFunction loss)
{
  long K = layer.output_size();
  long N = X.rows();

  // do a feedforward + backpropagate pass to compute Db and DW
  layer.X = X;
  eigen::matrix Y(K, N);
  layer.feedforward(Y);
  eigen::matrix DY = loss.gradient(Y, T);
  layer.backpropagate(Y, DY);

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
  long K = T.cols();
  long N = X.rows();

  // do a feedforward + backpropagate pass to compute Db and DW
  layer.X = X;
  eigen::matrix Y(K, N);
  layer.feedforward(Y);
  eigen::matrix DY = loss.gradient(Y, T);
  layer.backpropagate(Y, DY);

  auto f = [&]()
  {
    layer.feedforward(Y);
    return loss(Y, T);
  };

  scalar h = std::is_same<scalar, double>::value ? scalar(0.0001) : scalar(0.01);
  CHECK(check_gradient("Dbeta", f, layer.beta, layer.Dbeta, h));
  CHECK(check_gradient("Dgamma", f, layer.gamma, layer.Dgamma, h));
}

void test_mlp(multilayer_perceptron& M, const eigen::matrix& X, const eigen::matrix& T, std::shared_ptr<loss_function> loss)
{
  long N = X.rows();
  long K = T.cols();
  eigen::matrix Y(N, K);

  // do a feedforward + backpropagate pass to compute Db and DW
  M.layers.front()->X = X;
  M.feedforward(Y);
  eigen::matrix DY = loss->gradient(Y, T);
  M.backpropagate(Y, DY);

  auto f = [&]()
  {
    M.feedforward(Y);
    return loss->value(Y, T);
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

template <typename LossFunction>
void test_linear_layer(long D, long K, long N, LossFunction loss)
{
  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix Y = eigen::random_matrix(N, K);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);
  eigen::matrix W = eigen::random_matrix(K, D);
  eigen::matrix b = eigen::random_matrix(1, K);

  {
    linear_layer<eigen::matrix> layer(D, K, N);
    layer.W = W;
    layer.b = b;
    test_linear_layer(layer, X, T, loss);
  }

  {
    relu_layer<eigen::matrix> layer(D, K, N);
    layer.W = W;
    layer.b = b;
    test_linear_layer(layer, X, T, loss);
  }

  {
    sigmoid_layer<eigen::matrix> layer(D, K, N);
    layer.W = W;
    layer.b = b;
    test_linear_layer(layer, X, T, loss);
  }

  {
    softmax_layer<eigen::matrix> layer(D, K, N);
    layer.W = W;
    layer.b = b;
    test_linear_layer(layer, X, T, loss);
  }
}

TEST_CASE("test_linear_layer")
{
  std::vector<std::tuple<long, long, long>> parameters = { {3, 2, 2}, {2, 3, 2}, {2, 2, 3} };
  for (const auto& param : parameters)
  {
    auto [D, K, N] = param;
    test_linear_layer(D, K, N, squared_error_loss());
    test_linear_layer(D, K, N, cross_entropy_loss());
    test_linear_layer(D, K, N, logistic_cross_entropy_loss());
    test_linear_layer(D, K, N, softmax_cross_entropy_loss());
    test_linear_layer(D, K, N, negative_log_likelihood_loss());
  }
}

template <typename LossFunction>
void test_dropout_layer(long D, long K, long N, scalar p, LossFunction loss)
{
  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);
  eigen::matrix W = eigen::random_matrix(K, D);
  eigen::matrix b = eigen::random_matrix(1, K);

  linear_dropout_layer<eigen::matrix> layer1(D, K, N, p);
  layer1.W = W;
  layer1.b = b;
  test_linear_layer(layer1, X, T, loss);

  relu_dropout_layer<eigen::matrix> layer2(D, K, N, p);
  layer2.W = W;
  layer2.b = b;
  test_linear_layer(layer2, X, T, loss);

  sigmoid_dropout_layer<eigen::matrix> layer3(D, K, N, p);
  layer3.W = W;
  layer3.b = b;
  test_linear_layer(layer3, X, T, loss);
}

TEST_CASE("test_dropout_layer")
{
  scalar p = 0.8;

  std::vector<std::tuple<long, long, long>> parameters = { {3, 2, 2}, {2, 3, 2}, {2, 2, 3} };
  for (const auto& param : parameters)
  {
    auto [D, K, N] = param;
    test_dropout_layer(D, K, N, p, squared_error_loss());
    test_dropout_layer(D, K, N, p, cross_entropy_loss());
    test_dropout_layer(D, K, N, p, logistic_cross_entropy_loss());
    test_dropout_layer(D, K, N, p, softmax_cross_entropy_loss());
    test_dropout_layer(D, K, N, p, negative_log_likelihood_loss());
  }
}

template <typename LossFunction>
void test_batch_normalization_layer(long D, long N, LossFunction loss)
{
  long K = D;
  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);
  eigen::matrix beta = eigen::random_matrix(1, K);
  eigen::matrix gamma = eigen::random_matrix(1, K);

  {
    affine_layer layer(D, N);
    layer.beta = beta;
    layer.gamma = gamma;
    test_affine_layer(layer, X, T, loss);
  }

  {
    batch_normalization_layer layer(D, N);
    layer.beta = beta;
    layer.gamma = gamma;
    test_affine_layer(layer, X, T, loss);
  }
}

TEST_CASE("test_batch_normalization_layer")
{
  std::vector<std::tuple<long, long>> parameters = { {2, 3}, {3, 2} };
  for (const auto& param : parameters)
  {
    auto [D, N] = param;
    test_batch_normalization_layer(D, N, squared_error_loss());
    test_batch_normalization_layer(D, N, logistic_cross_entropy_loss());
    test_batch_normalization_layer(D, N, softmax_cross_entropy_loss());
    // test_batch_normalization_layer(D, N,  cross_entropy_loss());
    // test_batch_normalization_layer(D, N, negative_log_likelihood_loss());
  }
}

inline
void construct_mlp(multilayer_perceptron& M,
                   const std::vector<std::string>& layer_specifications,
                   const std::vector<std::size_t>& linear_layer_sizes,
                   const std::vector<double>& linear_layer_densities,
                   const std::vector<double>& linear_layer_dropouts,
                   const std::vector<std::string>& linear_layer_weights,
                   const std::vector<std::string>& optimizers,
                   long batch_size
                  )
{
  M.layers = make_layers(layer_specifications, linear_layer_sizes, linear_layer_densities, linear_layer_dropouts, linear_layer_weights, optimizers, batch_size, nerva_rng);
}

TEST_CASE("test_mlp0")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 5; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();

  construct_mlp(M,
                {"ReLU", "ReLU", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {"XavierNormalized", "XavierNormalized", "XavierNormalized"},
                {"GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
                );
  print_mlp("test_mlp0", M);

  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);

  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}

TEST_CASE("test_dropout_relu")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 3; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();

  construct_mlp(M,
                {"ReLU", "ReLU", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {0.5, 0.0, 0.0},
                {"XavierNormalized", "XavierNormalized", "XavierNormalized"},
                {"GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
  );
  renew_dropout_masks(M, nerva_rng);  // TODO: is this needed?

  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);

  print_mlp("test_dropout_relu", M);
  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}

TEST_CASE("test_dropout_linear")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 3; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();
  double p = 0.8;

  construct_mlp(M,
                {"ReLU", "ReLU", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {0.0, 0.0, p},
                {"Xavier", "Xavier", "Xavier"},
                {"GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
  );
  renew_dropout_masks(M, nerva_rng);  // TODO: is this needed?

  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);

  print_mlp("test_dropout_linear", M);
  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}

TEST_CASE("test_dropout_sigmoid")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 3; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();
  double p = 0.8;

  construct_mlp(M,
                {"Sigmoid", "Sigmoid", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {p, 0.0, 0.0},
                {"Xavier", "Xavier", "Xavier"},
                {"GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
  );
  renew_dropout_masks(M, nerva_rng);  // TODO: is this needed?

  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);

  print_mlp("test_dropout_sigmoid", M);
  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}

TEST_CASE("test_batch_normalization1")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 4; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();

  construct_mlp(M,
                {"BatchNorm", "ReLU", "ReLU", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {"Xavier", "Xavier", "Xavier"},
                {"GradientDescent", "GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
  );

  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);

  print_mlp("test_batch_normalization1", M);
  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}

TEST_CASE("test_batch_normalization2")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 4; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();

  construct_mlp(M,
                {"ReLU", "BatchNorm", "ReLU", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {"Xavier", "Xavier", "Xavier"},
                {"GradientDescent", "GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
  );

  eigen::matrix X = eigen::random_matrix(N, D);
  eigen::matrix T = eigen::random_target_rowwise(N, K, nerva_rng);

  print_mlp("test_batch_normalization2", M);
  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}

/* TODO
TEST_CASE("test_chessboard")
{
  multilayer_perceptron M;
  std::vector<std::size_t> linear_layer_sizes = {2, 2, 2, 2};
  long N = 3; // the number of examples
  long batch_size = N;
  long D = linear_layer_sizes.front();
  long K = linear_layer_sizes.back();

  construct_mlp(M,
                {"ReLU", "ReLU", "Linear"},
                linear_layer_sizes,
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {"Xavier", "Xavier", "Xavier"},
                {"GradientDescent", "GradientDescent", "GradientDescent"},
                batch_size
  );

  auto [X, T] = datasets::make_dataset_chessboard(N, nerva_rng);

  print_mlp("test_chessboard", M);
  std::vector<std::string> loss_functions = {"SquaredError", "LogisticCrossEntropy", "SoftmaxCrossEntropy"};
  for (const std::string& loss_function_text: loss_functions)
  {
    std::cout << "loss = " << loss_function_text << std::endl;
    std::shared_ptr<loss_function> loss = parse_loss_function(loss_function_text);
    test_mlp(M, X, T, loss);
  }
}
*/

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
    eigen::matrix Sigma = R.array().square().rowwise().sum() / N;
    eigen::matrix Y = eigen::Diag(eigen::inv_sqrt(Sigma)) * R;
    return fY(Y);
  };

  auto dfR = [dfY](const eigen::matrix& R)
  {
    using eigen::diag;
    using eigen::Diag;

    long N = R.cols();
    eigen::matrix Sigma = R.array().square().rowwise().sum() / N;
    eigen::matrix Y = Diag(eigen::inv_sqrt(Sigma)) * R;
    eigen::matrix DY = dfY(Y);
    eigen::matrix DiagDYY = Diag(diag(DY * Y.transpose())) * Y;
    eigen::matrix DR = Diag(eigen::inv_sqrt(Sigma)) * (-DiagDYY / N + DY);
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

TEST_CASE("test_inv_sqrt")
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

  eigen::matrix Y = Diag(eigen::inv_sqrt(Sigma)) * X;
  std::cout << "Y=\n" << Y << std::endl;

  CHECK_LE((expected - Y).squaredNorm(), 1e-10);
}
