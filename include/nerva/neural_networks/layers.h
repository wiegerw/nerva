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

  [[nodiscard]] virtual std::string name() const = 0;

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
      mkl::assign_matrix_product(result, W, X);
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
      mkl::assign_matrix_product_batch(DW, DY, X.transpose(), std::max(4L, static_cast<long>(DY.rows() / 10)));
      Db = DY.rowwise().sum();
      mkl::assign_matrix_product(DX, W, DY, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
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

  [[nodiscard]] std::string name() const override
  {
    return "linear layer";
  }

  void info(unsigned int layer_index) const override
  {
    std::string i = std::to_string(layer_index);
    std::cout << "---" << name() << " ---" << std::endl;
    print_numpy_matrix("W" + i, W);
    if constexpr (IsSparse)
    {
      print_numpy_matrix("stencil", mkl::stencil(W));
    }
    print_numpy_matrix("b" + i, b);
  }

  void reset_support()
  {
    if constexpr (IsSparse)
    {
      this->DW.reset_support(this->W);
      this->optimizer->reset_support(this->W);
    }
  }

  void import_weights(const eigen::matrix& W)
  {
    if constexpr (IsSparse)
    {
      compare_sizes(this->W, W);
      this->W = mkl::to_csr(W);
      reset_support();
    }
    else
    {
      compare_sizes(this->W, W);
      this->W = W;
    }
  }
};

template <typename Scalar>
void initialize_sparse_weights(linear_layer<mkl::sparse_matrix_csr<Scalar>>& layer, double density, std::mt19937& rng)
{
  auto m = layer.W.rows();
  auto n = layer.W.cols();
  layer.W = mkl::sparse_matrix_csr<Scalar>(m, n, density, rng, scalar(0));
  layer.DW = layer.W;
}

using dense_linear_layer = linear_layer<eigen::matrix>;
using sparse_linear_layer = linear_layer<mkl::sparse_matrix_csr<scalar>>;

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
      mkl::assign_matrix_product(Z, W, X);
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
      mkl::assign_matrix_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::assign_matrix_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = DY.cwiseProduct(eigen::x_times_one_minus_x(Y));
      DW = DZ * X.transpose();
      Db = DZ.rowwise().sum();
      DX = W.transpose() * DZ;
    }
  }

  [[nodiscard]] std::string name() const override
  {
    return "sigmoid layer";
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
      mkl::assign_matrix_product(Z, W, X);
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
      mkl::assign_matrix_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::assign_matrix_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
    }
    else
    {
      DZ = DY.cwiseProduct(act.prime(Z));
      DW = DZ * X.transpose();
      Db = DZ.rowwise().sum();
      DX = W.transpose() * DZ;
    }
  }

  [[nodiscard]] std::string name() const override
  {
    return "activation layer";
  }
};

template <typename Matrix>
struct hyperbolic_tangent_layer : public activation_layer<Matrix, hyperbolic_tangent_activation>
{
  using super = activation_layer<Matrix, hyperbolic_tangent_activation>;

  explicit hyperbolic_tangent_layer(std::size_t D, std::size_t K, std::size_t Q = 1)
   : super(hyperbolic_tangent_activation(), D, K, Q)
  {}

  [[nodiscard]] std::string name() const override
  {
    return "hyperbolic_tangent layer";
  }
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

  [[nodiscard]] std::string name() const override
  {
    return "relu layer";
  }
};

using dense_relu_layer = relu_layer<eigen::matrix>;
using sparse_relu_layer = relu_layer<mkl::sparse_matrix_csr<scalar>>;

template <typename Matrix>
struct leaky_relu_layer : public activation_layer<Matrix, leaky_relu_activation>
{
  using super = activation_layer<Matrix, leaky_relu_activation>;

  explicit leaky_relu_layer(scalar alpha, std::size_t D, std::size_t K, std::size_t Q = 1)
      : super(leaky_relu_activation(alpha), D, K, Q)
  {}

  [[nodiscard]] std::string name() const override
  {
    return "leaky relu layer";
  }
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

  [[nodiscard]] std::string name() const override
  {
    return "all relu layer";
  }
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
      mkl::assign_matrix_product(Z, W, X);
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
      mkl::assign_matrix_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::assign_matrix_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
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

  [[nodiscard]] std::string name() const override
  {
    return "softmax layer";
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
      mkl::assign_matrix_product(Z, W, X);
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
      mkl::assign_matrix_product_batch(DW, DZ, X.transpose(), std::max(4L, static_cast<long>(DZ.rows() / 10)));
      Db = DZ.rowwise().sum();
      mkl::assign_matrix_product(DX, W, DZ, scalar(0), scalar(1), SPARSE_OPERATION_TRANSPOSE);
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

  [[nodiscard]] std::string name() const override
  {
    return "log softmax layer";
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
      throw std::runtime_error("Error: could not parse optimizer '" + text + "'");
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

  std::size_t total_params = 0;
  for (const auto& [rows, columns]: layer_shapes)
  {
    total_params += rows * columns;
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
      auto n_param = rows * columns;
      auto n_zeros = n_param * (double(1) - overall_density);
      auto n_ones = n_param * overall_density;
      if (dense_layers.count(i))
      {
        rhs -= n_zeros;
      }
      else
      {
        rhs += n_ones;
        raw_probabilities[i] = ((rows + columns) / (double)(rows * columns)) * std::pow(erk_power_scale, double(1));
        divisor += raw_probabilities[i] * n_param;
      }
    }
    epsilon = rhs / divisor;
    double max_prob = *std::max_element(raw_probabilities.begin(), raw_probabilities.end());
    double max_prob_one = max_prob * epsilon;
    if (max_prob_one > 1)
    {
      for (auto j = 0; j < n; j++)
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
  double total_nonzero = 0;
  for (auto i = 0; i < n; i++)
  {
    auto rows = layer_shapes[i].first;
    auto columns = layer_shapes[i].second;
    auto n_param = rows * columns;
    if (dense_layers.count(i))
    {
      densities[i] = 1;
    }
    else
    {
      double probability_one = epsilon * raw_probabilities[i];
      densities[i] = probability_one;
    }
    total_nonzero += densities[i] * n_param;
  }
  return densities;
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LAYERS_H
