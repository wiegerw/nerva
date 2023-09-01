// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/batch_normalization_layers_rowwise.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/layers_rowwise.h"
#include "fmt/format.h"
#include <random>

namespace nerva::rowwise {

struct batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix Z;
  eigen::matrix DZ;
  eigen::matrix gamma;
  eigen::matrix Dgamma;
  eigen::matrix beta;
  eigen::matrix Dbeta;
  eigen::matrix power_minus_half_Sigma;
  std::shared_ptr<optimizer_function> optimizer;

  explicit batch_normalization_layer(std::size_t D, std::size_t N = 1)
   : super(D, N), Z(N, D), DZ(N, D), gamma(1, D), Dgamma(1, D), beta(1, D), Dbeta(1, D), power_minus_half_Sigma(1, D)
  {
    beta.array() = 0;
    gamma.array() = 1;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("BatchNormalization(input_size={}, output_size={})", Z.rows(), Z.rows());
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::power_minus_half;
    using eigen::row_repeat;
    using eigen::columns_mean;
    auto N = X.rows();

    auto R = (X - row_repeat(columns_mean(X), N)).eval();
    auto Sigma = diag(R.transpose() * R).transpose() / N;
    power_minus_half_Sigma = power_minus_half(Sigma);
    Z = hadamard(row_repeat(power_minus_half_Sigma, N), R);
    result = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::identity;
    using eigen::ones;
    using eigen::power_minus_half;
    auto N = X.rows();

    DZ = hadamard(row_repeat(gamma, N), DY);
    Dbeta = columns_sum(DY);
    Dgamma = columns_sum(hadamard(Z, DY));
    DX = hadamard(row_repeat(power_minus_half_Sigma / N, N), (N * identity<eigen::matrix>(N) - ones<eigen::matrix>(N, N)) * DZ - hadamard(Z, row_repeat(diag(Z.transpose() * DZ).transpose(), N)));
  }

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }
};

using dense_batch_normalization_layer = batch_normalization_layer;

// batch normalization without an affine transformation
struct simple_batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix power_minus_half_Sigma;
  std::shared_ptr<optimizer_function> optimizer;

  explicit simple_batch_normalization_layer(std::size_t D, std::size_t N = 1)
    : super(D, N), power_minus_half_Sigma(1, D)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return "SimpleBatchNormalization()";
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::power_minus_half;
    using eigen::row_repeat;
    using eigen::columns_mean;
    auto N = X.rows();

    auto R = (X - row_repeat(columns_mean(X), N)).eval();
    auto Sigma = diag(R.transpose() * R).transpose() / N;
    power_minus_half_Sigma = power_minus_half(Sigma);
    result = hadamard(row_repeat(power_minus_half_Sigma, N), R);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::identity;
    using eigen::ones;
    using eigen::power_minus_half;
    using eigen::column_repeat;

    auto N = X.rows();
    DX = hadamard(column_repeat(power_minus_half_Sigma / N, N), hadamard(Y, column_repeat(-diag(DY * Y.transpose()), N)) + DY * (N * identity<eigen::matrix>(N) - ones<eigen::matrix>(N, N)));
  }

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }
};

using dense_simple_batch_normalization_layer = simple_batch_normalization_layer;

struct affine_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix gamma;
  eigen::matrix Dgamma;
  eigen::matrix beta;
  eigen::matrix Dbeta;
  std::shared_ptr<optimizer_function> optimizer;

  explicit affine_layer(std::size_t D, std::size_t N = 1)
    : super(D, N), gamma(1, D), Dgamma(1, D), beta(1, D), Dbeta(1, D)
  {
    beta.array() = 0;
    gamma.array() = 1;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "Affine()";
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::hadamard;
    using eigen::row_repeat;
    auto N = X.rows();

    result = hadamard(row_repeat(gamma, N), X) + row_repeat(beta, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::row_repeat;
    using eigen::columns_sum;
    auto N = X.rows();

    DX = hadamard(row_repeat(gamma, N), DY);
    Dbeta = columns_sum(DY);
    Dgamma = columns_sum(hadamard(X, DY));
  }

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }
};

using dense_affine_layer = affine_layer;

template <typename BatchNormalizationLayer>
void set_batch_normalization_layer_optimizer(BatchNormalizationLayer& layer, const std::string& text)
{
  auto optimizer_beta = parse_optimizer(text, layer.beta, layer.Dbeta);
  auto optimizer_gamma = parse_optimizer(text, layer.gamma, layer.Dgamma);
  layer.optimizer = make_composite_optimizer(optimizer_beta, optimizer_gamma);
}

} // namespace nerva::rowwise

#include "nerva/neural_networks/rowwise_colwise.inc"
