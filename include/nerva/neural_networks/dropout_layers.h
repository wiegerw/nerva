// Copyright: Wieger Wesselink 2022
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

  linear_dropout_layer(std::size_t D, std::size_t K, std::size_t Q, scalar p)
   : super(D, K, Q), dropout_layer<Matrix>(D, K, p)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dropout({})\n{}", p, super::to_string());
  }

  void feedforward(eigen::matrix& result) override
  {
    auto Q = X.cols();
    result = W.cwiseProduct(R) * X + b.rowwise().replicate(Q);
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

  sigmoid_dropout_layer(std::size_t D, std::size_t K, std::size_t Q, scalar p)
      : super(D, K, Q), dropout_layer<Matrix>(D, K, p)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dropout({})\n{}", p, super::to_string());
  }

  void feedforward(eigen::matrix& result) override
  {
    auto Q = X.cols();
    Z = W.cwiseProduct(R) * X + b.rowwise().replicate(Q);
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

  activation_dropout_layer(ActivationFunction act, std::size_t D, std::size_t K, std::size_t Q, scalar p)
      : super(act, D, K, Q), dropout_layer<Matrix>(D, K, p)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Dropout({})\n{}", p, super::to_string());
  }

  void feedforward(eigen::matrix& result) override
  {
    auto Q = X.cols();
    Z = W.cwiseProduct(R) * X + b.rowwise().replicate(Q);
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

  relu_dropout_layer(std::size_t D, std::size_t K, std::size_t Q, scalar p)
   : super(relu_activation(), D, K, Q, p)
  {}
};

using dense_relu_dropout_layer = relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct all_relu_dropout_layer: public activation_dropout_layer<Matrix, all_relu_activation>
{
  using super = activation_dropout_layer<Matrix, all_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  all_relu_dropout_layer(scalar alpha, std::size_t D, std::size_t K, std::size_t Q, scalar p)
   : super(all_relu_activation(alpha), D, K, Q, p)
  {}
};

using dense_all_relu_dropout_layer = all_relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct leaky_relu_dropout_layer: public activation_dropout_layer<Matrix, leaky_relu_activation>
{
  using super = activation_dropout_layer<Matrix, leaky_relu_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  leaky_relu_dropout_layer(scalar alpha, std::size_t D, std::size_t K, std::size_t Q, scalar p)
      : super(leaky_relu_activation(alpha), D, K, Q, p)
  {}
};

using dense_leaky_relu_dropout_layer = leaky_relu_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct hyperbolic_tangent_dropout_layer: public activation_dropout_layer<Matrix, hyperbolic_tangent_activation>
{
  using super = activation_dropout_layer<Matrix, hyperbolic_tangent_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  hyperbolic_tangent_dropout_layer(std::size_t D, std::size_t K, std::size_t Q, scalar p)
      : super(hyperbolic_tangent_activation(), D, K, Q, p)
  {}
};

using dense_hyperbolic_tangent_dropout_layer = hyperbolic_tangent_dropout_layer<eigen::matrix>;

template <typename Matrix>
struct softmax_dropout_layer: public activation_dropout_layer<Matrix, softmax_activation>
{
  using super = activation_dropout_layer<Matrix, softmax_activation>;
  using super::to_string;
  using dropout_layer<Matrix>::p;

  softmax_dropout_layer(std::size_t D, std::size_t K, std::size_t Q, scalar p)
      : super(softmax_activation(), D, K, Q, p)
  {}
};

using dense_softmax_dropout_layer = softmax_dropout_layer<eigen::matrix>;

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_DROPOUT_LAYERS_H
