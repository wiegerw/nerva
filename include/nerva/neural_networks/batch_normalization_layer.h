// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/batch_normalization_layer.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_BATCH_NORMALIZATION_LAYER_H
#define NERVA_NEURAL_NETWORKS_BATCH_NORMALIZATION_LAYER_H

#include "nerva/neural_networks/layers.h"
#include <random>

namespace nerva {

struct batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix Z;
  eigen::matrix DZ;
  eigen::vector gamma;
  eigen::vector Dgamma;
  eigen::vector beta;
  eigen::vector Dbeta;
  eigen::vector Sigma;
  eigen::matrix R;

  explicit batch_normalization_layer(std::size_t D, std::size_t N = 1)
   : super(D, N), Z(D, N), DZ(D, N), gamma(D, 1), Dgamma(D, 1), beta(D, 1), Dbeta(D, 1), Sigma(D, 1), R(D, N)
  {
    beta.array() = 0;
    gamma.array() = 1;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "BatchNormalization()";
  }

  void optimize(scalar eta) override
  {
    // TODO use an optimizer as in linear_layer?
    beta -= eta * Dbeta;
    gamma -= eta * Dgamma;
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::power_minus_half;
    using eigen::repeat_column;
    using eigen::rowwise_mean;

    auto N = X.cols();
    R = X - repeat_column(rowwise_mean(X), N);
    Sigma = diag(R * R.transpose()) / N;
    Z = hadamard(repeat_column(power_minus_half(Sigma), N), R);
    result = hadamard(repeat_column(gamma, N), Z) + repeat_column(beta, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::repeat_column;
    using eigen::sum_rows;
    using eigen::identity;
    using eigen::ones;
    using eigen::power_minus_half;

    auto N = X.cols();
    Dbeta = sum_rows(DY);
    Dgamma = hadamard(DY, Z).rowwise().sum();
    DZ = hadamard(repeat_column(gamma, N), DY);

    // TODO: attempts to reuse the computation power_minus_half(Sigma) make the code run slower with g++-12. Why?
    auto Sigma_power_minus_half = power_minus_half(Sigma);
    DX = hadamard(repeat_column(Sigma_power_minus_half / N, N), hadamard(Z, repeat_column(-diag(DZ * Z.transpose()), N)) + DZ * (N * identity<eigen::matrix>(N) - ones<eigen::matrix>(N, N)));
  }
};

using dense_batch_normalization_layer = batch_normalization_layer;

// batch normalization without an affine transformation
struct simple_batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix R;
  eigen::vector Sigma;

  explicit simple_batch_normalization_layer(std::size_t D, std::size_t N = 1)
    : super(D, N), R(D, N), Sigma(D, 1)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return "SimpleBatchNormalization()";
  }

  void optimize(scalar eta) override
  {}

  void feedforward(eigen::matrix& result) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::power_minus_half;
    using eigen::repeat_column;
    using eigen::rowwise_mean;

    auto N = X.cols();
    R = X - repeat_column(rowwise_mean(X), N);
    Sigma = diag(R * R.transpose()) / N;
    result = hadamard(repeat_column(power_minus_half(Sigma), N), R);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::identity;
    using eigen::ones;
    using eigen::power_minus_half;
    using eigen::repeat_column;

    auto N = X.cols();
    auto Sigma_power_minus_half = power_minus_half(Sigma);
    DX = hadamard(repeat_column(Sigma_power_minus_half / N, N), hadamard(Y, repeat_column(-diag(DY * Y.transpose()), N)) + DY * (N * identity<eigen::matrix>(N) - ones<eigen::matrix>(N, N)));
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

  explicit affine_layer(std::size_t D, std::size_t N = 1)
    : super(D, N), gamma(D, 1), Dgamma(D, 1), beta(D, 1), Dbeta(D, 1)
  {
    beta.array() = 0;
    gamma.array() = 1;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "Affine()";
  }

  void optimize(scalar eta) override
  {
    // TODO use an optimizer as in linear_layer?
    beta -= eta * Dbeta;
    gamma -= eta * Dgamma;
  }

  void feedforward(eigen::matrix& result) override
  {
    using eigen::hadamard;
    using eigen::repeat_column;

    auto N = X.cols();
    result = hadamard(repeat_column(gamma, N), X) + repeat_column(beta, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::repeat_column;
    using eigen::sum_rows;

    auto N = X.cols();
    DX = hadamard(repeat_column(gamma, N), DY);
    Dbeta = sum_rows(DY);
    Dgamma = sum_rows(hadamard(X, DY));
  }
};

using dense_affine_layer = affine_layer;

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_BATCH_NORMALIZATION_LAYER_H
