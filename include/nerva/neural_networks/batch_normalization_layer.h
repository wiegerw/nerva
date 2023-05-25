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

template <typename Matrix>
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
  eigen::vector Sigma_power_minus_half;
  eigen::vector diag_DZ_Zt;

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
    using eigen::repeat_column;
    using eigen::hadamard;

    auto N = X.cols();
    scalar epsilon = 1e-20;
    R = X.colwise() - X.rowwise().mean();
    Sigma = R.array().square().rowwise().sum() / N;
    Z = hadamard(repeat_column(eigen::power_minus_half(Sigma, epsilon), N), R);
    result = hadamard(repeat_column(gamma, N), Z) + repeat_column(beta, N);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::repeat_column;
    using eigen::sum_rows;

    auto N = X.cols();
    scalar epsilon = 1e-20;
    Dbeta = sum_rows(DY);
    Dgamma = hadamard(DY, Z).rowwise().sum();
    DZ = hadamard(repeat_column(gamma, N), DY);

    // TODO: attempts to reuse the computation power_minus_half(Sigma, epsilon) make the code run slower with g++-12. Why?
    Sigma_power_minus_half = eigen::power_minus_half(Sigma, epsilon);
    diag_DZ_Zt = diag(DZ * Z.transpose()) / N;
    DX = hadamard(repeat_column(Sigma_power_minus_half, N), (hadamard(repeat_column(-diag_DZ_Zt, N), Z) + DZ * (eigen::matrix::Identity(N, N) - eigen::matrix::Constant(N, N, scalar(1) / N))));
  }
};

using dense_batch_normalization_layer = batch_normalization_layer<eigen::matrix>;

// batch normalization without an affine transformation
template <typename Matrix>
struct simple_batch_normalization_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;

  eigen::matrix R;
  eigen::vector Sigma;
  eigen::vector Sigma_power_minus_half;
  eigen::vector diag_DY_Yt;

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
    using eigen::hadamard;
    using eigen::repeat_column;

    auto N = X.cols();
    scalar epsilon = 1e-20;
    R = X.colwise() - X.rowwise().mean();
    Sigma = R.array().square().rowwise().sum() / N;
    result = hadamard(repeat_column(eigen::power_minus_half(Sigma, epsilon), N), R);
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::Diag;
    using eigen::hadamard;
    using eigen::repeat_column;

    auto N = X.cols();
    scalar epsilon = 1e-20;
    Sigma_power_minus_half = eigen::power_minus_half(Sigma, epsilon) / N;  // N.B. Also divide by N for efficiency reasons
    diag_DY_Yt = diag(DY * Y.transpose());
    DX = hadamard(repeat_column(Sigma_power_minus_half, N), (Diag(-diag_DY_Yt) * Y + DY * (N * eigen::matrix::Identity(N, N) - eigen::matrix::Constant(N, N, scalar(1)))));
  }
};

using dense_simple_batch_normalization_layer = simple_batch_normalization_layer<eigen::matrix>;

template <typename Matrix>
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
    Dgamma = hadamard(DY, X).rowwise().sum();
  }
};

using dense_affine_layer = affine_layer<eigen::matrix>;

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_BATCH_NORMALIZATION_LAYER_H
