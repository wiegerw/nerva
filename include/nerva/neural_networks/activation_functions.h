// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/snn/activation_functions.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_ACTIVATION_FUNCTIONS_H
#define NERVA_NEURAL_NETWORKS_ACTIVATION_FUNCTIONS_H

#include "nerva/neural_networks/eigen.h"
#include <cmath>
#include <ratio>

namespace nerva {

struct sigmoid
{
  scalar operator()(scalar x) const
  {
    return scalar(1.0) / (scalar(1.0) + std::exp(-x));
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct sigmoid_prime
{
  scalar operator()(scalar x) const
  {
    scalar f_x = sigmoid()(x);
    return f_x * (scalar(1.0) - f_x);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct sigmoid_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return sigmoid()(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return sigmoid_prime()(X);
  }
};

struct relu
{
  scalar operator()(scalar x) const
  {
    return std::max(scalar(0.0), x);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct relu_prime
{
  scalar operator()(scalar x) const
  {
    return (x < 0) ? scalar(0.0) : scalar(1.0);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct relu_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return relu()(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return relu_prime()(X);
  }
};

struct leaky_relu
{
  scalar alpha;

  explicit leaky_relu(scalar alpha_)
    : alpha(alpha_)
  {}

  scalar operator()(scalar x) const
  {
    return std::max(alpha*x, x);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct leaky_relu_prime
{
  scalar alpha;

  explicit leaky_relu_prime(scalar alpha_)
      : alpha(alpha_)
  {}

  scalar operator()(scalar x) const
  {
    return (x < 0) ? alpha : scalar(1.0);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct leaky_relu_activation
{
  scalar alpha;

  explicit leaky_relu_activation(scalar alpha_)
      : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return leaky_relu(alpha)(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return leaky_relu_prime(alpha)(X);
  }
};

struct all_relu
{
  scalar alpha;

  explicit all_relu(scalar alpha_)
      : alpha(alpha_)
  {}

  scalar operator()(scalar x) const
  {
    return x < scalar(0) ? alpha * x : x;
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct all_relu_prime
{
  scalar alpha;

  explicit all_relu_prime(scalar alpha_)
      : alpha(alpha_)
  {}

  scalar operator()(scalar x) const
  {
    return x < scalar(0) ? alpha : scalar(1);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

// combine all_relu and all_relu_prime
struct all_relu_activation
{
  scalar alpha;

  explicit all_relu_activation(scalar alpha_)
   : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return all_relu(alpha)(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return all_relu_prime(alpha)(X);
  }
};

struct hyperbolic_tangent
{
  scalar operator()(scalar x) const
  {
    return std::tanh(x);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct hyperbolic_tangent_prime
{
  // https://stackoverflow.com/questions/24511540/how-to-calculate-the-derivative-of-fnet-tanhnet-in-c
  scalar operator()(scalar x) const
  {
    scalar tanh_x = hyperbolic_tangent()(x);
    return scalar(1.0) - tanh_x * tanh_x;
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct hyperbolic_tangent_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return hyperbolic_tangent()(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return hyperbolic_tangent_prime()(X);
  }
};

struct softmax
{
  eigen::vector operator()(const eigen::vector& x) const
  {
    // use the log-sum-exp trick to make the computation robust, see also https://en.wikipedia.org/wiki/LogSumExp
    scalar c = x.maxCoeff();
    auto E = ((x.array() - c)).exp();
    return E / E.sum();
  }

  // see also https://gist.github.com/WilliamTambellini/8294f211800e16791d47f3cf59472a49
  eigen::matrix operator()(const eigen::matrix& X) const
  {
    auto c = X.colwise().maxCoeff().eval();
    auto x_minus_c = X.rowwise() - c;
    auto E = x_minus_c.array().exp();
    return E.rowwise() / E.colwise().sum();
  }

  eigen::matrix log(const eigen::matrix& X)
  {
    auto c = X.colwise().maxCoeff().eval();
    auto x_minus_c = X.rowwise() - c;
    auto E = x_minus_c.array().exp();
    return x_minus_c.array().rowwise() - E.colwise().sum().log();
  }
};

struct softmax_prime
{
  eigen::vector operator()(const eigen::vector& x) const
  {
    return softmax()(x).unaryExpr([](scalar t) { return t * (scalar(1) - t); });
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    return X.diagonal() - X * X.transpose();
  }
};

struct softmax_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return softmax()(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return softmax_prime()(X);
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_ACTIVATION_FUNCTIONS_H
