// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/layers_rowwise.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LAYERS_ROWWISE_H
#define NERVA_NEURAL_NETWORKS_LAYERS_ROWWISE_H

#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/layer_algorithms.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/optimizers.h"
#include "nerva/neural_networks/softmax_rowwise.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/string_utility.h"
#include "fmt/format.h"
#include <iostream>
#include <random>
#include <type_traits>

namespace nerva {

// prototypes
template <typename Matrix> struct linear_layer;
template <typename Matrix> struct dropout_layer;
template <typename Matrix> struct batch_normalization_layer;

struct neural_network_layer
{
  eigen::matrix X;  // the input
  eigen::matrix DX; // the gradient of the input

  explicit neural_network_layer(std::size_t D, std::size_t N)
    : X(D, N), DX(D, N)
  {}

  [[nodiscard]] virtual std::string to_string() const = 0;

  virtual void optimize(scalar eta) = 0;

  virtual void feedforward(eigen::matrix& result) = 0;

  virtual void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) = 0;

  virtual void info(unsigned int layer_index) const
  {}

  virtual ~neural_network_layer() = default;
};

template <typename Matrix>
struct linear_layer: public neural_network_layer
{
  using super = neural_network_layer;
  using super::X;
  using super::DX;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  Matrix W;
  eigen::vector b;
  Matrix DW;
  eigen::vector Db;
  std::shared_ptr<layer_optimizer> optimizer;

  explicit linear_layer(std::size_t D, std::size_t K, std::size_t N)
    : super(D, N), W(K, D), b(K), DW(K, D), Db(K)
  {}

  [[nodiscard]] std::size_t input_size() const
  {
    return W.cols();
  }

  [[nodiscard]] std::size_t output_size() const
  {
    return W.rows();
  }

  [[nodiscard]] std::string to_string() const override
  {
    if constexpr (IsSparse)
    {
      return fmt::format("Sparse(units={}, density={}, optimizer={}, activation=NoActivation())", output_size(), W.density(), optimizer->to_string());
    }
    else
    {
      return fmt::format("Dense(units={}, optimizer={}, activation=NoActivation())", output_size(), optimizer->to_string());
    }
  }

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(result, W, X);
      result += repeat_column(b, N);
    }
    else
    {
      auto N = X.cols();
      result = W * X + repeat_column(b, N);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::sum_rows;

    if constexpr (IsSparse)
    {
      mkl::sdd_product_batch(DW, DY, X.transpose(), std::max(4L, static_cast<long>(DY.rows() / 10)));
      Db = sum_rows(DY);
      mkl::dsd_product(DX, W, DY, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DW = DY * X.transpose();
      Db = sum_rows(DY);
      DX = W.transpose() * DY;
    }
  }

  void optimize(scalar eta) override
  {
    optimizer->update(eta);
  }

  void info(unsigned int layer_index) const override
  {
    std::string i = std::to_string(layer_index);
    std::cout << to_string() << std::endl;
    print_numpy_matrix("W" + i, W);
    if constexpr (IsSparse)
    {
      print_numpy_matrix("support", mkl::support(W));
    }
    print_numpy_matrix("b" + i, b);
  }

  void reset_support()
  {
    if constexpr (IsSparse)
    {
      DW.reset_support(W);
      if (optimizer)
      {
        optimizer->reset_support();
      }
    }
  }

  void load_weights(const Matrix& W1)
  {
    compare_sizes(W, W1);
    W = W1;
    reset_support();
  }
};

using dense_linear_layer = linear_layer<eigen::matrix>;
using sparse_linear_layer = linear_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Scalar>
void set_support_random(linear_layer<mkl::sparse_matrix_csr<Scalar>>& layer, double density, std::mt19937& rng)
{
  auto rows = layer.W.rows();
  auto columns = layer.W.cols();
  std::size_t size = std::lround(density * rows * columns);
  layer.W = mkl::make_random_matrix<Scalar>(rows, columns, size, rng);
  layer.reset_support();
}

template <typename Matrix>
void set_weights_and_bias(linear_layer<Matrix>& layer, weight_initialization w, std::mt19937& rng)
{
  auto& W = layer.W;
  auto init = make_weight_initializer(w, W, rng);
  set_weights(W, [&init]() { return (*init)(); });
  init->initialize_bias(layer.b);
}

template <typename Matrix>
struct sigmoid_layer : public linear_layer<Matrix>
{
  using super = linear_layer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  using super::optimizer;
  using super::output_size;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  eigen::matrix Z;
  eigen::matrix DZ;

  sigmoid_layer(std::size_t D, std::size_t K, std::size_t N)
    : super(D, K, N), Z(K, N), DZ(K, N)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    if constexpr (IsSparse)
    {
      return fmt::format("Sparse(units={}, density={}, optimizer={}, activation=Sigmoid())", output_size(), W.density(), optimizer->to_string());
    }
    else
    {
      return fmt::format("Dense(units={}, optimizer={}, activation=Sigmoid())", output_size(), optimizer->to_string());
    }
  }

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += repeat_column(b, N);
      result = sigmoid()(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + repeat_column(b, N);
      result = sigmoid()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::sum_rows;

    if constexpr (IsSparse)
    {
      DZ = hadamard(DY, eigen::x_times_one_minus_x(Y));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = sum_rows(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = hadamard(DY, eigen::x_times_one_minus_x(Y));
      DW = DZ * X.transpose();
      Db = sum_rows(DZ);
      DX = W.transpose() * DZ;
    }
  }
};

using dense_sigmoid_layer = sigmoid_layer<eigen::matrix>;
using sparse_sigmoid_layer = sigmoid_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix, typename ActivationFunction>
struct activation_layer : public linear_layer<Matrix>
{
  using super = linear_layer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  using super::optimizer;
  using super::output_size;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  ActivationFunction act;
  eigen::matrix Z;
  eigen::matrix DZ;

  explicit activation_layer(std::size_t D, std::size_t K, std::size_t N, ActivationFunction act_)
    : super(D, K, N), act(act_), Z(K, N), DZ(K, N)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    if constexpr (IsSparse)
    {
      return fmt::format("Sparse(units={}, density={}, optimizer={}, activation={})", output_size(), W.density(), optimizer->to_string(), act.to_string());
    }
    else
    {
      return fmt::format("Dense(units={}, optimizer={}, activation={})", output_size(), optimizer->to_string(), act.to_string());
    }
  }

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += repeat_column(b, N);
      result = act(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + repeat_column(b, N);
      result = act(Z);
    }
  }

  void backpropagate(const eigen::matrix& /* Y */, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::sum_rows;

    if constexpr (IsSparse)
    {
      DZ = hadamard(DY, act.prime(Z));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      // mkl::sdd_product(DW, DZ, X.transpose());
      Db = sum_rows(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = hadamard(DY, act.prime(Z));
      DW = DZ * X.transpose();
      Db = sum_rows(DZ);
      DX = W.transpose() * DZ;
    }
  }
};

template <typename Matrix>
struct hyperbolic_tangent_layer : public activation_layer<Matrix, hyperbolic_tangent_activation>
{
  using super = activation_layer<Matrix, hyperbolic_tangent_activation>;

  explicit hyperbolic_tangent_layer(std::size_t D, std::size_t K, std::size_t N)
    : super(D, K, N, hyperbolic_tangent_activation())
  {}
};

using dense_hyperbolic_tangent_layer = hyperbolic_tangent_layer<eigen::matrix>;
using sparse_hyperbolic_tangent_layer = hyperbolic_tangent_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct relu_layer : public activation_layer<Matrix, relu_activation>
{
  using super = activation_layer<Matrix, relu_activation>;

  explicit relu_layer(std::size_t D, std::size_t K, std::size_t N)
    : super(D, K, N, relu_activation())
  {}
};

using dense_relu_layer = relu_layer<eigen::matrix>;
using sparse_relu_layer = relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct trelu_layer : public activation_layer<Matrix, trimmed_relu_activation>
{
  using super = activation_layer<Matrix, trimmed_relu_activation>;

  explicit trelu_layer(std::size_t D, std::size_t K, std::size_t N, scalar epsilon)
    : super(D, K, N, trimmed_relu_activation(epsilon))
  {}
};

using dense_trelu_layer = trelu_layer<eigen::matrix>;
using sparse_trelu_layer = trelu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct leaky_relu_layer : public activation_layer<Matrix, leaky_relu_activation>
{
  using super = activation_layer<Matrix, leaky_relu_activation>;

  explicit leaky_relu_layer(std::size_t D, std::size_t K, std::size_t N, scalar alpha)
    : super(D, K, N, leaky_relu_activation(alpha))
  {}
};

using dense_leaky_relu_layer = leaky_relu_layer<eigen::matrix>;
using sparse_leaky_relu_layer = leaky_relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct all_relu_layer : public activation_layer<Matrix, all_relu_activation>
{
  using super = activation_layer<Matrix, all_relu_activation>;

  explicit all_relu_layer(std::size_t D, std::size_t K, std::size_t N, scalar alpha)
    : super(D, K, N, all_relu_activation(alpha))
  {}
};

using dense_all_relu_layer = all_relu_layer<eigen::matrix>;
using sparse_all_relu_layer = all_relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct srelu_layer : public activation_layer<Matrix, srelu_activation>
{
  using super = activation_layer<Matrix, srelu_activation>;
  using super::backpropagate;
  using super::optimize;
  using super::Z;
  using super::act;

  scalar Dal = 0;
  scalar Dtl = 0;
  scalar Dar = 0;
  scalar Dtr = 0;

  explicit srelu_layer(std::size_t D, std::size_t K, std::size_t N, scalar al = 1, scalar tl = 0, scalar ar = 1, scalar tr = 0)
    : super(D, K, N, srelu_activation(al, tl, ar, tr))
  {}

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    super::backpropagate(Y, DY);

    auto A_l = Z.unaryExpr([this](scalar x) { return std::min(x - act.tl, scalar(0)); });
    Dal = hadamard(DY, A_l).sum();

    auto A_r = Z.unaryExpr([this](scalar x) { return std::max(x - act.tr, scalar(0)); });
    Dar = hadamard(DY, A_r).sum();

    auto T_l = Z.unaryExpr([this](scalar x) { return x <= act.tl ? scalar(1) - act.al : scalar(0); });
    Dtl = hadamard(DY, T_l).sum();

    auto T_r = Z.unaryExpr([this](scalar x) { return x >= act.tr ? scalar(1) - act.ar : scalar(0); });
    Dtr = hadamard(DY, T_r).sum();
  }

  void optimize(scalar eta) override
  {
    super::optimize(eta);
    act.al -= eta * Dal;
    act.ar -= eta * Dar;
    act.tl -= eta * Dtl;
    act.tr -= eta * Dtr;
  }
};

using dense_srelu_layer = srelu_layer<eigen::matrix>;
using sparse_srelu_layer = srelu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct softmax_layer : public linear_layer<Matrix>
{
  using super = linear_layer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  eigen::matrix Z;
  eigen::matrix DZ;

  softmax_layer(std::size_t D, std::size_t K, std::size_t N)
    : super(D, K, N), Z(K, N), DZ(K, N)
  {}

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += repeat_column(b, N);
      result = softmax_colwise()(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + repeat_column(b, N);
      result = softmax_colwise()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::repeat_row;
    using eigen::sum_rows;

    if constexpr (IsSparse)
    {
      auto K = Y.rows();
      DZ = hadamard(Y, DY - repeat_row(diag(Y.transpose() * DY).transpose(), K));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = sum_rows(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      auto K = Y.rows();
      DZ = hadamard(Y, DY - repeat_row(diag(Y.transpose() * DY).transpose(), K));
      DW = DZ * X.transpose();
      Db = sum_rows(DZ);
      DX = W.transpose() * DZ;
    }
  }
};

using dense_softmax_layer = softmax_layer<eigen::matrix>;
using sparse_softmax_layer = softmax_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct log_softmax_layer : public linear_layer<Matrix>
{
  using super = linear_layer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  using super::X;
  using super::DX;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  eigen::matrix Z;
  eigen::matrix DZ;

  log_softmax_layer(std::size_t D, std::size_t K, std::size_t N)
    : super(D, K, N), Z(K, N), DZ(K, N)
  {}

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += repeat_column(b, N);
      result = log_softmax_colwise()(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + repeat_column(b, N);
      result = log_softmax_colwise()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::repeat_row;
    using eigen::sum_rows;

    if constexpr (IsSparse)
    {
      auto K = Y.rows();
      DZ = DY - repeat_row(diag(Y.transpose() * DY).transpose(), K);
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = sum_rows(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      auto K = Y.rows();
      DZ = DY - repeat_row(diag(Y.transpose() * DY), K);
      DW = DZ * X.transpose();
      Db = sum_rows(DZ);
      DX = W.transpose() * DZ;
    }
  }
};

using dense_log_softmax_layer = log_softmax_layer<eigen::matrix>;
using sparse_log_softmax_layer = log_softmax_layer<mkl::sparse_matrix_csr<scalar>>;

// Sets the optimizer in a layer
template <typename Matrix>
void set_optimizer(linear_layer<Matrix>& layer, const std::string& text)
{
  auto parse_argument = [&text]()
  {
    auto startpos = text.find('(');
    auto endpos = text.find(')');
    if (startpos == std::string::npos || endpos == std::string::npos || endpos <= startpos)
    {
      throw std::runtime_error("could not parse optimizer '" + text + "'");
    }
    return parse_scalar(text.substr(startpos + 1, endpos - startpos - 1));
  };

  if (text == "GradientDescent")
  {
    layer.optimizer = std::make_shared<gradient_descent_optimizer<Matrix>>(layer.W, layer.DW, layer.b, layer.Db);
  }
  else if (utilities::starts_with(text, "Momentum"))  // e.g. "momentum(0.9)"
  {
    scalar mu = parse_argument();
    layer.optimizer = std::make_shared<momentum_optimizer<Matrix>>(layer.W, layer.DW, layer.b, layer.Db, mu);
  }
  else if (utilities::starts_with(text, "Nesterov"))
  {
    scalar mu = parse_argument();
    layer.optimizer = std::make_shared<nesterov_optimizer<Matrix>>(layer.W, layer.DW, layer.b, layer.Db, mu);
  }
  else
  {
    throw std::runtime_error("unknown optimizer '" + text + "'");
  }
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LAYERS_ROWWISE_H
