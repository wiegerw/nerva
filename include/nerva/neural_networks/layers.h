// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/layers.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LAYERS_H
#define NERVA_NEURAL_NETWORKS_LAYERS_H

#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/optimizers.h"
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

inline
void compare_sizes(const eigen::matrix& W1, const eigen::matrix& W2)
{
  if (W1.rows() != W2.rows() || W1.cols() != W2.cols())
  {
    eigen::print_numpy_matrix("W", W1);
    eigen::print_numpy_matrix("W", W2);
    throw std::runtime_error("matrix sizes do not match");
  }
}

inline
void compare_sizes(const mkl::sparse_matrix_csr<scalar>& W1, const eigen::matrix& W2)
{
  if (W1.rows() != W2.rows() || W1.cols() != W2.cols())
  {
    eigen::print_numpy_matrix("W", mkl::to_eigen(W1));
    eigen::print_numpy_matrix("W", W2);
    throw std::runtime_error("matrix sizes do not match");
  }
}

struct neural_network_layer
{
  eigen::matrix X;  // the input
  eigen::matrix DX; // the gradient of the input

  explicit neural_network_layer(std::size_t D, std::size_t Q = 1)
   : X(D, Q), DX(D, Q)
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

  explicit linear_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
   : super(D, Q), W(K, D), b(K), DW(K, D), Db(K)
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
      auto Q = X.cols();
      mkl::dsd_product(result, W, X);
      result += b.rowwise().replicate(Q);
    }
    else
    {
      auto Q = X.cols();
      result = W * X + b.rowwise().replicate(Q);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    if constexpr (IsSparse)
    {
      mkl::sdd_product_batch(DW, DY, X.transpose(), std::max(4L, static_cast<long>(DY.rows() / 10)));
      Db = DY.rowwise().sum();
      mkl::dsd_product(DX, W, DY, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DW = DY * X.transpose();
      Db = DY.rowwise().sum();
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

  sigmoid_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
      : super(D, K, Q), Z(K, Q), DZ(K, Q)
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
      auto Q = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += b.rowwise().replicate(Q);
      result = sigmoid()(Z);
    }
    else
    {
      auto Q = X.cols();
      Z = W * X + b.rowwise().replicate(Q);
      result = sigmoid()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    if constexpr (IsSparse)
    {
      DZ = DY.cwiseProduct(eigen::x_times_one_minus_x(Y));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = DY.cwiseProduct(eigen::x_times_one_minus_x(Y));
      DW = DZ * X.transpose();
      Db = DZ.rowwise().sum();
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

  explicit activation_layer(ActivationFunction act_, std::size_t D, std::size_t K, std::size_t Q = 1)
   : super(D, K, Q), act(act_), Z(K, Q), DZ(K, Q)
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
      auto Q = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += b.rowwise().replicate(Q);
      result = act(Z);
    }
    else
    {
      auto Q = X.cols();
      Z = W * X + b.rowwise().replicate(Q);
      result = act(Z);
    }
  }

  void backpropagate(const eigen::matrix& /* Y */, const eigen::matrix& DY) override
  {
    if constexpr (IsSparse)
    {
      DZ = DY.cwiseProduct(act.prime(Z));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      // mkl::sdd_product(DW, DZ, X.transpose());
      Db = DZ.rowwise().sum();
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = DY.cwiseProduct(act.prime(Z));
      DW = DZ * X.transpose();
      Db = DZ.rowwise().sum();
      DX = W.transpose() * DZ;
    }
  }
};

template <typename Matrix>
struct hyperbolic_tangent_layer : public activation_layer<Matrix, hyperbolic_tangent_activation>
{
  using super = activation_layer<Matrix, hyperbolic_tangent_activation>;

  explicit hyperbolic_tangent_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
   : super(hyperbolic_tangent_activation(), D, K, Q)
  {}
};

using dense_hyperbolic_tangent_layer = hyperbolic_tangent_layer<eigen::matrix>;
using sparse_hyperbolic_tangent_layer = hyperbolic_tangent_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct relu_layer : public activation_layer<Matrix, relu_activation>
{
  using super = activation_layer<Matrix, relu_activation>;

  explicit relu_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
      : super(relu_activation(), D, K, Q)
  {}
};

using dense_relu_layer = relu_layer<eigen::matrix>;
using sparse_relu_layer = relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct trimmed_relu_layer : public activation_layer<Matrix, trimmed_relu_activation>
{
  using super = activation_layer<Matrix, trimmed_relu_activation>;

  explicit trimmed_relu_layer(scalar epsilon, std::size_t D, std::size_t K, std::size_t Q = 1)
    : super(trimmed_relu_activation(epsilon), D, K, Q)
  {}
};

using dense_trimmed_relu_layer = trimmed_relu_layer<eigen::matrix>;
using sparse_trimmed_relu_layer = trimmed_relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct leaky_relu_layer : public activation_layer<Matrix, leaky_relu_activation>
{
  using super = activation_layer<Matrix, leaky_relu_activation>;

  explicit leaky_relu_layer(scalar alpha, std::size_t D, std::size_t K, std::size_t Q = 1)
      : super(leaky_relu_activation(alpha), D, K, Q)
  {}
};

using dense_leaky_relu_layer = leaky_relu_layer<eigen::matrix>;
using sparse_leaky_relu_layer = leaky_relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct all_relu_layer : public activation_layer<Matrix, all_relu_activation>
{
  using super = activation_layer<Matrix, all_relu_activation>;

  explicit all_relu_layer(scalar alpha, std::size_t D, std::size_t K, std::size_t Q = 1)
      : super(all_relu_activation(alpha), D, K, Q)
  {}
};

using dense_all_relu_layer = all_relu_layer<eigen::matrix>;
using sparse_all_relu_layer = all_relu_layer<mkl::sparse_matrix_csr<scalar>>;

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

  softmax_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
      : super(D, K, Q), Z(K, Q), DZ(K, Q)
  {}

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto Q = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += b.rowwise().replicate(Q);
      result = softmax()(Z);
    }
    else
    {
      auto Q = X.cols();
      Z = W * X + b.rowwise().replicate(Q);
      result = softmax()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    if constexpr (IsSparse)
    {
      auto K = Y.rows();
      DZ = Y.cwiseProduct(DY - (Y.transpose() * DY).diagonal().transpose().colwise().replicate(K));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      auto K = Y.rows();
      DZ = Y.cwiseProduct(DY - (Y.transpose() * DY).diagonal().transpose().colwise().replicate(K));
      DW = DZ * X.transpose();
      Db = DZ.rowwise().sum();
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

  log_softmax_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
    : super(D, K, Q), Z(K, Q), DZ(K, Q)
  {}

  void feedforward(eigen::matrix& result) override
  {
    if constexpr (IsSparse)
    {
      auto Q = X.cols();
      mkl::dsd_product(Z, W, X);
      Z += b.rowwise().replicate(Q);
      result = log_softmax()(Z);
    }
    else
    {
      auto Q = X.cols();
      Z = W * X + b.rowwise().replicate(Q);
      result = log_softmax()(Z);
    }
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY) override
  {
    if constexpr (IsSparse)
    {
      auto K = Y.rows();
      auto softmax_Z = softmax()(Z);
      DZ = DY - softmax_Z.cwiseProduct(DY.colwise().sum().colwise().replicate(K));
      mkl::sdd_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::dsd_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      auto K = Y.rows();
      auto softmax_Z = softmax()(Z);
      DZ = DY - softmax_Z.cwiseProduct(DY.colwise().sum().colwise().replicate(K));
      DW = DZ * X.transpose();
      Db = DZ.rowwise().sum();
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

  if (text == "gradient-descent")
  {
    layer.optimizer = std::make_shared<gradient_descent_optimizer<Matrix>>(layer.W, layer.DW, layer.b, layer.Db);
  }
  else if (utilities::starts_with(text, "momentum"))  // e.g. "momentum(0.9)"
  {
    scalar mu = parse_argument();
    layer.optimizer = std::make_shared<momentum_optimizer<Matrix>>(layer.W, layer.DW, layer.b, layer.Db, mu);
  }
  else if (utilities::starts_with(text, "nesterov"))
  {
    scalar mu = parse_argument();
    layer.optimizer = std::make_shared<nesterov_optimizer<Matrix>>(layer.W, layer.DW, layer.b, layer.Db, mu);
  }
  else
  {
    throw std::runtime_error("unknown optimizer '" + text + "'");
  }
}

inline
std::vector<double> compute_sparse_layer_densities(double overall_density,
                                                   const std::vector<std::size_t>& layer_sizes,
                                                   double erk_power_scale = 1
                                                  )
{
  std::vector<std::pair<std::size_t, std::size_t>> layer_shapes;
  for (std::size_t i = 0; i < layer_sizes.size() - 1; i++)
  {
    layer_shapes.emplace_back(layer_sizes[i], layer_sizes[i+1]);
  }

  std::size_t n = layer_shapes.size(); // the number of layers

  if (overall_density == 1)
  {
    return std::vector<double>(n, 1);
  }

  std::set<std::size_t> dense_layers;
  std::vector<double> raw_probabilities(n, double(0));
  double epsilon;

  while (true)
  {
    double divisor = 0;
    double rhs = 0;
    std::fill(raw_probabilities.begin(), raw_probabilities.end(), double(0));
    for (std::size_t i = 0; i < n; i++)
    {
      auto [rows, columns] = layer_shapes[i];
      auto N = rows * columns;
      auto num_ones = N * overall_density;
      auto num_zeros = N - num_ones;
      if (dense_layers.count(i))
      {
        rhs -= num_zeros;
      }
      else
      {
        rhs += num_ones;
        raw_probabilities[i] = ((rows + columns) / (double)(rows * columns)) * std::pow(erk_power_scale, double(1));
        divisor += raw_probabilities[i] * N;
      }
    }
    epsilon = rhs / divisor;
    double max_prob = *std::max_element(raw_probabilities.begin(), raw_probabilities.end());
    double max_prob_one = max_prob * epsilon;
    if (max_prob_one > 1)
    {
      for (std::size_t j = 0; j < n; j++)
      {
        if (raw_probabilities[j] == max_prob)
        {
          dense_layers.insert(j);
        }
      }
    }
    else
    {
      break;
    }
  }

  // Compute the densities
  std::vector<double> densities(n, 0);
  for (std::size_t i = 0; i < n; i++)
  {
    if (dense_layers.count(i))
    {
      densities[i] = 1;
    }
    else
    {
      double probability_one = epsilon * raw_probabilities[i];
      densities[i] = probability_one;
    }
  }
  return densities;
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LAYERS_H
