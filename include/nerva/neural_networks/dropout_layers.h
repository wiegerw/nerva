// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/dropout_layers.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_DROPOUT_LAYERS_H
#define NERVA_NEURAL_NETWORKS_DROPOUT_LAYERS_H

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
  using super::to_string;
  using dropout_layer<Matrix>::p;
  using dropout_layer<Matrix>::R;

  linear_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
   : super(D, K, N), dropout_layer<Matrix>(D, K, p)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dropout({})\n{}", p, super::to_string());
  }

  void feedforward(eigen::matrix& result) override
  {
    auto N = X.cols();
    result = W.cwiseProduct(R) * X + b.rowwise().replicate(N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    if constexpr (std::is_same<Matrix, eigen::matrix>::value)
    {
      DW = (DY * X.transpose()).cwiseProduct(R);
    }
    else
    {
      // TODO
    }
    Db = DY.rowwise().sum();
    DX = W.cwiseProduct(R).transpose() * DY;
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
  using super::to_string;
  using dropout_layer<Matrix>::p;
  using dropout_layer<Matrix>::R;

  sigmoid_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
      : super(D, K, N), dropout_layer<Matrix>(D, K, p)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dropout({})\n{}", p, super::to_string());
  }

  void feedforward(eigen::matrix& result) override
  {
    auto N = X.cols();
    Z = W.cwiseProduct(R) * X + b.rowwise().replicate(N);
    result = sigmoid()(Z);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    DZ = DY.cwiseProduct(eigen::x_times_one_minus_x(Y));
    if constexpr (std::is_same<Matrix, eigen::matrix>::value)
    {
      DW = (DZ * X.transpose()).cwiseProduct(R);
    }
    else
    {
      // TODO
    }
    Db = DZ.rowwise().sum();
    DX = W.cwiseProduct(R).transpose() * DZ;
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
  using super::to_string;
  using dropout_layer<Matrix>::p;
  using dropout_layer<Matrix>::R;
  using super::act;

  activation_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, ActivationFunction act)
      : super(D, K, N, act), dropout_layer<Matrix>(D, K, p)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dropout({})\n{}", p, super::to_string());
  }

  void feedforward(eigen::matrix& result) override
  {
    auto N = X.cols();
    Z = W.cwiseProduct(R) * X + b.rowwise().replicate(N);
    result = act(Z);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    DZ = DY.cwiseProduct(act.prime(Z));
    if constexpr (std::is_same<Matrix, eigen::matrix>::value)
    {
      DW = (DZ * X.transpose()).cwiseProduct(R);
    }
    else
    {
      // TODO
    }
    Db = DZ.rowwise().sum();
    DX = W.cwiseProduct(R).transpose() * DZ;
  }
};

template <typename Matrix>
struct relu_dropout_layer: public activation_dropout_layer<Matrix, relu_activation>
{
  using super = activation_dropout_layer<Matrix, relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  relu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
   : super(D, K, N, p, relu_activation())
  {}
};
using dense_relu_dropout_layer = relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct softmax_dropout_layer: public activation_dropout_layer<Matrix, softmax_colwise_activation>
{
  using super = activation_dropout_layer<Matrix, softmax_colwise_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  softmax_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
    : super(D, K, N, p, softmax_colwise_activation())
  {}
};
using dense_softmax_dropout_layer = softmax_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct log_softmax_dropout_layer: public activation_dropout_layer<Matrix, log_softmax_colwise_activation>
{
  using super = activation_dropout_layer<Matrix, log_softmax_colwise_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  log_softmax_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
    : super(D, K, N, p, log_softmax_colwise_activation())
  {}
};
using dense_log_softmax_dropout_layer = log_softmax_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct hyperbolic_tangent_dropout_layer: public activation_dropout_layer<Matrix, hyperbolic_tangent_activation>
{
  using super = activation_dropout_layer<Matrix, hyperbolic_tangent_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  hyperbolic_tangent_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p)
      : super(D, K, N, p, hyperbolic_tangent_activation())
  {}
};
using dense_hyperbolic_tangent_dropout_layer = hyperbolic_tangent_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct all_relu_dropout_layer: public activation_dropout_layer<Matrix, all_relu_activation>
{
  using super = activation_dropout_layer<Matrix, all_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  all_relu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar alpha)
    : super(D, K, N, p, all_relu_activation(alpha))
  {}
};
using dense_all_relu_dropout_layer = all_relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct leaky_relu_dropout_layer: public activation_dropout_layer<Matrix, leaky_relu_activation>
{
  using super = activation_dropout_layer<Matrix, leaky_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  leaky_relu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar alpha)
    : super(D, K, N, p, leaky_relu_activation(alpha))
  {}
};
using dense_leaky_relu_dropout_layer = leaky_relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct trelu_dropout_layer: public activation_dropout_layer<Matrix, trimmed_relu_activation>
{
  using super = activation_dropout_layer<Matrix, trimmed_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  trelu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar epsilon)
    : super(D, K, N, p, trimmed_relu_activation(epsilon))
  {}
};
using dense_trelu_dropout_layer = trelu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct srelu_dropout_layer: public activation_dropout_layer<Matrix, srelu_activation>
{
  using super = activation_dropout_layer<Matrix, srelu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  srelu_dropout_layer(std::size_t D, std::size_t K, std::size_t N, scalar p, scalar al = 1, scalar tl = 0, scalar ar = 1, scalar tr = 0)
    : super(D, K, N, p, srelu_activation(al, tl, ar, tr))
  {}
};
using dense_srelu_dropout_layer = srelu_dropout_layer<eigen::matrix>;

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_DROPOUT_LAYERS_H
