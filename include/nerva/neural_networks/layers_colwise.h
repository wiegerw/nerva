// Copyright: Wieger Wesselink 2022 - present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/layers_colwise.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LAYERS_COLWISE_H
#define NERVA_NEURAL_NETWORKS_LAYERS_COLWISE_H

#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/layer_algorithms.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/optimizers.h"
#include "nerva/neural_networks/softmax_colwise.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/string_utility.h"
#include "fmt/format.h"
#include <iostream>
#include <random>
#include <type_traits>

namespace nerva {

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
  eigen::matrix b;
  Matrix DW;
  eigen::matrix Db;
  std::shared_ptr<optimizer_function> optimizer;

  explicit linear_layer(std::size_t D, std::size_t K, std::size_t N)
   : super(D, N), W(K, D), b(K, 1), DW(K, D), Db(K, 1)
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
    using eigen::column_repeat;

    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(result, W, X);
      result += column_repeat(b, N);
    }
    else
    {
      auto N = X.cols();
      result = W * X + column_repeat(b, N);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::rows_sum;

    if constexpr (IsSparse)
    {
      mkl::sdd_product_batch(DW, DY, X.transpose(), std::max(4L, static_cast<long>(DY.rows() / 10)));
      Db = rows_sum(DY);
      mkl::dsd_product(DX, W, DY, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DW = DY * X.transpose();
      Db = rows_sum(DY);
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
    using eigen::column_repeat;
    using eigen::Sigmoid;

    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += column_repeat(b, N);
      result = Sigmoid(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + column_repeat(b, N);
      result = Sigmoid(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::rows_sum;

    if constexpr (IsSparse)
    {
      DZ = hadamard(DY, eigen::x_times_one_minus_x(Y));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = rows_sum(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = hadamard(DY, eigen::x_times_one_minus_x(Y));
      DW = DZ * X.transpose();
      Db = rows_sum(DZ);
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
    using eigen::column_repeat;

    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += column_repeat(b, N);
      result = act(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + column_repeat(b, N);
      result = act(Z);
    }
  }

  void backpropagate(const eigen::matrix& /* Y */, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::rows_sum;

    if constexpr (IsSparse)
    {
      DZ = hadamard(DY, act.gradient(Z));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      // mkl::sdd_product(DW, DZ, X.transpose());
      Db = rows_sum(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = hadamard(DY, act.gradient(Z));
      DW = DZ * X.transpose();
      Db = rows_sum(DZ);
      DX = W.transpose() * DZ;
    }
  }
};

template <typename Matrix>
struct hyperbolic_tangent_layer : public activation_layer<Matrix, eigen::hyperbolic_tangent_activation>
{
  using super = activation_layer<Matrix, eigen::hyperbolic_tangent_activation>;

  explicit hyperbolic_tangent_layer(std::size_t D, std::size_t K, std::size_t N)
   : super(D, K, N, eigen::hyperbolic_tangent_activation())
  {}
};

using dense_hyperbolic_tangent_layer = hyperbolic_tangent_layer<eigen::matrix>;
using sparse_hyperbolic_tangent_layer = hyperbolic_tangent_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct relu_layer : public activation_layer<Matrix, eigen::relu_activation>
{
  using super = activation_layer<Matrix, eigen::relu_activation>;

  explicit relu_layer(std::size_t D, std::size_t K, std::size_t N)
      : super(D, K, N, eigen::relu_activation())
  {}
};

using dense_relu_layer = relu_layer<eigen::matrix>;
using sparse_relu_layer = relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct trelu_layer : public activation_layer<Matrix, eigen::trimmed_relu_activation>
{
  using super = activation_layer<Matrix, eigen::trimmed_relu_activation>;

  explicit trelu_layer(std::size_t D, std::size_t K, std::size_t N, scalar epsilon)
    : super(D, K, N, eigen::trimmed_relu_activation(epsilon))
  {}
};

using dense_trelu_layer = trelu_layer<eigen::matrix>;
using sparse_trelu_layer = trelu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct leaky_relu_layer : public activation_layer<Matrix, eigen::leaky_relu_activation>
{
  using super = activation_layer<Matrix, eigen::leaky_relu_activation>;

  explicit leaky_relu_layer(std::size_t D, std::size_t K, std::size_t N, scalar alpha)
      : super(D, K, N, eigen::leaky_relu_activation(alpha))
  {}
};

using dense_leaky_relu_layer = leaky_relu_layer<eigen::matrix>;
using sparse_leaky_relu_layer = leaky_relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct all_relu_layer : public activation_layer<Matrix, eigen::all_relu_activation>
{
  using super = activation_layer<Matrix, eigen::all_relu_activation>;

  explicit all_relu_layer(std::size_t D, std::size_t K, std::size_t N, scalar alpha)
      : super(D, K, N, eigen::all_relu_activation(alpha))
  {}
};

using dense_all_relu_layer = all_relu_layer<eigen::matrix>;
using sparse_all_relu_layer = all_relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct srelu_layer : public activation_layer<Matrix, eigen::srelu_activation>
{
  using super = activation_layer<Matrix, eigen::srelu_activation>;
  using super::backpropagate;
  using super::optimize;
  using super::Z;
  using super::act;

  explicit srelu_layer(std::size_t D, std::size_t K, std::size_t N, scalar al = 1, scalar tl = 0, scalar ar = 1, scalar tr = 0)
    : super(D, K, N, eigen::srelu_activation(al, tl, ar, tr))
  {}

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::apply;
    using eigen::hadamard;
    using eigen::elements_sum;

    super::backpropagate(Y, DY);

    auto al = act.x(0);
    auto tl = act.x(1);
    auto ar = act.x(2);
    auto tr = act.x(3);

    auto Al = [tl](scalar x)     { return x <= tl ? x - tl : scalar(0); };
    auto Ar = [tl, tr](scalar x) { return x <= tl || x < tr ? scalar(0) : x - tr; };
    auto Tl = [tl, al](scalar x) { return x <= tl ? scalar(1) - al : scalar(0); };
    auto Tr = [tr, ar](scalar x) { return x >= tr ? scalar(1) - ar : scalar(0); };

    auto Dal = elements_sum(hadamard(DY, apply(Al, Z)));
    auto Dar = elements_sum(hadamard(DY, apply(Ar, Z)));
    auto Dtl = elements_sum(hadamard(DY, apply(Tl, Z)));
    auto Dtr = elements_sum(hadamard(DY, apply(Tr, Z)));

    act.Dx = eigen::vector{{Dal, Dtl, Dar, Dtr}};
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
    using eigen::column_repeat;

    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += column_repeat(b, N);
      result = stable_softmax()(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + column_repeat(b, N);
      result = stable_softmax()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::diag;
    using eigen::hadamard;
    using eigen::row_repeat;
    using eigen::rows_sum;

    if constexpr (IsSparse)
    {
      auto K = Y.rows();
      DZ = eigen::hadamard(Y, DY - row_repeat(diag(Y.transpose() * DY).transpose(), K));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = rows_sum(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      auto K = Y.rows();
      DZ = eigen::hadamard(Y, DY - row_repeat(diag(Y.transpose() * DY).transpose(), K));
      DW = DZ * X.transpose();
      Db = rows_sum(DZ);
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
    using eigen::column_repeat;

    if constexpr (IsSparse)
    {
      auto N = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += column_repeat(b, N);
      result = stable_log_softmax()(Z);
    }
    else
    {
      auto N = X.cols();
      Z = W * X + column_repeat(b, N);
      result = stable_log_softmax()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    using eigen::hadamard;
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::rows_sum;

    if constexpr (IsSparse)
    {
      auto K = Y.rows();
      DZ = DY - eigen::hadamard(stable_softmax()(Z), row_repeat(columns_sum(DY), K));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = rows_sum(DZ);
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      auto K = Y.rows();
      DZ = DY - eigen::hadamard(stable_softmax()(Z), row_repeat(columns_sum(DY), K));
      DW = DZ * X.transpose();
      Db = rows_sum(DZ);
      DX = W.transpose() * DZ;
    }
  }
};

using dense_log_softmax_layer = log_softmax_layer<eigen::matrix>;
using sparse_log_softmax_layer = log_softmax_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
void set_linear_layer_optimizer(linear_layer<Matrix>& layer, const std::string& text)
{
  auto optimizer_W = parse_optimizer(text, layer.W, layer.DW);
  auto optimizer_b = parse_optimizer(text, layer.b, layer.Db);
  layer.optimizer = make_composite_optimizer(optimizer_W, optimizer_b);
}

template <typename Matrix>
void set_srelu_layer_optimizer(srelu_layer<Matrix>& layer, const std::string& text)
{
  auto optimizer_W = parse_optimizer(text, layer.W, layer.DW);
  auto optimizer_b = parse_optimizer(text, layer.b, layer.Db);
  auto optimizer_srelu = parse_optimizer(text, layer.act.x, layer.act.Dx);
  layer.optimizer = make_composite_optimizer(optimizer_W, optimizer_b, optimizer_srelu);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LAYERS_COLWISE_H
