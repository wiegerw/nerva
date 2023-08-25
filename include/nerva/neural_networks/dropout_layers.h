// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/dropout_layers.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/layers.h"
#include "fmt/format.h"
#include <random>

namespace nerva {

template <typename Matrix>
struct dropout_layer
{
  Matrix R;
  scalar p;

  dropout_layer(std::size_t D, std::size_t K, scalar p_)
   : p(p_)
  {
    R = eigen::matrix::Constant(K, D, scalar(1));
  }

  void renew(std::mt19937& rng)
  {
    std::bernoulli_distribution dist(p);
    R = R.unaryExpr([&](scalar X) { return dist(rng) / p; });
  }
};

template <typename Matrix>
struct linear_dropout_layer: public linear_layer<Matrix>, dropout_layer<Matrix>
{
  using super = linear_layer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  using super::optimizer;
  using super::to_string;
  using super::input_size;
  using super::output_size;
  using dropout_layer<Matrix>::p;
  using dropout_layer<Matrix>::R;

  linear_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
   : super(D, K, N), dropout_layer<Matrix>(D, K, p)
  {}

  void feedforward(eigen::matrix& result) override
  {
    using eigen::column_repeat;
    using eigen::hadamard;

    auto N = X.cols();
    result = hadamard(W, R) * X + column_repeat(b, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::rows_sum;

    if constexpr (std::is_same<Matrix, eigen::matrix>::value)
    {
      DW = hadamard(DY * X.transpose(), R);
    }
    else
    {
      // TODO
    }
    Db = rows_sum(DY);
    DX = hadamard(W, R).transpose() * DY;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dense(input_size={}, output_size={}, optimizer={}, activation=NoActivation(), dropout={})", input_size(), output_size(), optimizer->to_string(), p);
  }
};

using dense_linear_dropout_layer = linear_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct sigmoid_dropout_layer: public sigmoid_layer<Matrix>, dropout_layer<Matrix>
{
  using super = sigmoid_layer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  using super::Z;
  using super::DZ;
  using super::optimizer;
  using super::input_size;
  using super::output_size;
  using super::to_string;
  using dropout_layer<Matrix>::p;
  using dropout_layer<Matrix>::R;

  sigmoid_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
      : super(D, K, N), dropout_layer<Matrix>(D, K, p)
  {}

  void feedforward(eigen::matrix& result) override
  {
    using eigen::column_repeat;
    using eigen::hadamard;
    using eigen::Sigmoid;

    auto N = X.cols();
    Z = hadamard(W, R) * X + column_repeat(b, N);
    result = Sigmoid(Z);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::rows_sum;

    DZ = hadamard(DY, eigen::x_times_one_minus_x(Y));
    if constexpr (std::is_same<Matrix, eigen::matrix>::value)
    {
      DW = hadamard(DZ * X.transpose(), R);
    }
    else
    {
      // TODO
    }
    Db = rows_sum(DZ);
    DX = hadamard(W, R).transpose() * DZ;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dense(input_size={}, output_size={}, optimizer={}, activation=Sigmoid(), dropout={})", input_size(), output_size(), optimizer->to_string(), p);
  }
};

using dense_sigmoid_dropout_layer = sigmoid_dropout_layer<eigen::matrix>;

template <typename Matrix, typename ActivationFunction>
struct activation_dropout_layer: public activation_layer<Matrix, ActivationFunction>, dropout_layer<Matrix>
{
  using super = activation_layer<Matrix, ActivationFunction>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  using super::Z;
  using super::DZ;
  using super::optimizer;
  using super::input_size;
  using super::output_size;
  using super::to_string;
  using dropout_layer<Matrix>::p;
  using dropout_layer<Matrix>::R;
  using super::act;

  activation_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, ActivationFunction act)
      : super(D, K, N, act), dropout_layer<Matrix>(D, K, p)
  {}

  void feedforward(eigen::matrix& result) override
  {
    using eigen::column_repeat;
    using eigen::hadamard;

    auto N = X.cols();
    Z = hadamard(W, R) * X + column_repeat(b, N);
    result = act(Z);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::rows_sum;

    DZ = hadamard(DY, act.gradient(Z));
    if constexpr (std::is_same<Matrix, eigen::matrix>::value)
    {
      DW = hadamard(DZ * X.transpose(), R);
    }
    else
    {
      // TODO
    }
    Db = rows_sum(DZ);
    DX = hadamard(W, R).transpose() * DZ;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dense(input_size={}, output_size={}, optimizer={}, activation={}, dropout={})", input_size(), output_size(), optimizer->to_string(), act.to_string(), p);
  }
};

template <typename Matrix>
struct relu_dropout_layer: public activation_dropout_layer<Matrix, eigen::relu_activation>
{
  using super = activation_dropout_layer<Matrix, eigen::relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  relu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
   : super(D, K, N, p, eigen::relu_activation())
  {}
};
using dense_relu_dropout_layer = relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct softmax_dropout_layer: public softmax_layer<Matrix>, dropout_layer<Matrix>
{
  using super = softmax_layer<Matrix>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  softmax_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
    : super(D, K, N), dropout_layer<Matrix>(D, K, p)
  {}
};
using dense_softmax_dropout_layer = softmax_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct log_softmax_dropout_layer: public log_softmax_layer<Matrix>, dropout_layer<Matrix>
{
  using super = log_softmax_layer<Matrix>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  log_softmax_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
    : super(D, K, N), dropout_layer<Matrix>(D, K, p)
  {}
};
using dense_log_softmax_dropout_layer = log_softmax_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct hyperbolic_tangent_dropout_layer: public activation_dropout_layer<Matrix, eigen::hyperbolic_tangent_activation>
{
  using super = activation_dropout_layer<Matrix, eigen::hyperbolic_tangent_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  hyperbolic_tangent_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
      : super(D, K, N, p, eigen::hyperbolic_tangent_activation())
  {}
};
using dense_hyperbolic_tangent_dropout_layer = hyperbolic_tangent_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct all_relu_dropout_layer: public activation_dropout_layer<Matrix, eigen::all_relu_activation>
{
  using super = activation_dropout_layer<Matrix, eigen::all_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  all_relu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar alpha)
    : super(D, K, N, p, eigen::all_relu_activation(alpha))
  {}
};
using dense_all_relu_dropout_layer = all_relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct leaky_relu_dropout_layer: public activation_dropout_layer<Matrix, eigen::leaky_relu_activation>
{
  using super = activation_dropout_layer<Matrix, eigen::leaky_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  leaky_relu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar alpha)
    : super(D, K, N, p, eigen::leaky_relu_activation(alpha))
  {}
};
using dense_leaky_relu_dropout_layer = leaky_relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct trelu_dropout_layer: public activation_dropout_layer<Matrix, eigen::trimmed_relu_activation>
{
  using super = activation_dropout_layer<Matrix, eigen::trimmed_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  trelu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar epsilon)
    : super(D, K, N, p, eigen::trimmed_relu_activation(epsilon))
  {}
};
using dense_trelu_dropout_layer = trelu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct srelu_dropout_layer: public activation_dropout_layer<Matrix, eigen::srelu_activation>
{
  using super = activation_dropout_layer<Matrix, eigen::srelu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  srelu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar al = 1, scalar tl = 0, scalar ar = 1, scalar tr = 0)
    : super(D, K, N, p, eigen::srelu_activation(al, tl, ar, tr))
  {}
};
using dense_srelu_dropout_layer = srelu_dropout_layer<eigen::matrix>;

} // namespace nerva

