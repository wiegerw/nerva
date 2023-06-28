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
#include "fmt/format.h"
#include <cmath>
#include <ratio>
#include <string>

namespace nerva {

namespace abc {

inline
scalar relu(scalar x)
{
  return std::max(scalar(0), x);
}

inline
scalar relu_derivative(scalar x)
{
  return (x < 0) ? scalar(0) : scalar(1);
}

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
};

struct leaky_relu_derivative
{
  scalar alpha;

  explicit leaky_relu_derivative(scalar alpha_)
    : alpha(alpha_)
  {}

  scalar operator()(scalar x) const
  {
    return (x < 0) ? alpha : scalar(1.0);
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
};

struct all_relu_derivative
{
  scalar alpha;

  explicit all_relu_derivative(scalar alpha_)
    : alpha(alpha_)
  {}

  scalar operator()(scalar x) const
  {
    return x < scalar(0) ? alpha : scalar(1);
  }
};

inline
scalar hyperbolic_tangent(scalar x)
{
  return std::tanh(x);
}

inline
scalar hyperbolic_tangent_derivative(scalar x)
{
  scalar y = hyperbolic_tangent(x);
  return scalar(1.0) - y * y;
}

inline
scalar sigmoid(scalar x)
{
  return scalar(1.0) / (scalar(1.0) + std::exp(-x));
}

inline
scalar sigmoid_derivative(scalar x)
{
  scalar y = sigmoid(x);
  return y * (scalar(1.0) - y);
}

struct srelu
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit srelu(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
    : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  scalar operator()(scalar x) const
  {
    if (x <= tl)
    {
      return tl + al * (x - tl);
    }
    else if (x < tr)
    {
      return x;
    }
    else
    {
      return tr + ar * (x - tr);
    }
  }
};

struct srelu_derivative
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit srelu_derivative(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
    : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  scalar operator()(scalar x) const
  {
    if (x <= tl)
    {
      return al;
    }
    else if (x < tr)
    {
      return 1;
    }
    else
    {
      return ar;
    }
  }
};

struct trimmed_relu
{
  scalar epsilon;

  explicit trimmed_relu(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  scalar operator()(scalar x) const
  {
    return (x < epsilon) ? scalar(0) : x;
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct trimmed_relu_derivative
{
  scalar epsilon;

  explicit trimmed_relu_derivative(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  scalar operator()(scalar x) const
  {
    return (x < epsilon) ? scalar(0) : scalar(1);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

/////////////////////////////////////////////////////////////////////////////////////

template <typename Matrix>
auto Relu(const Matrix& X)
{
  return eigen::apply(relu, X);
}

template <typename Matrix>
auto Relu_gradient(const Matrix& X)
{
  return eigen::apply(relu_derivative, X);
}

struct Leaky_relu
{
  scalar alpha;

  Leaky_relu(scalar alpha_)
    : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X)
  {
    return eigen::apply(leaky_relu(alpha), X);
  }
};

struct Leaky_relu_gradient
{
  scalar alpha;

  Leaky_relu_gradient(scalar alpha_)
    : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X)
  {
    return eigen::apply(leaky_relu_derivative(alpha), X);
  }
};

struct All_relu
{
  scalar alpha;

  All_relu(scalar alpha_)
    : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X)
  {
    return eigen::apply(all_relu(alpha), X);
  }
};

struct All_relu_gradient
{
  scalar alpha;

  All_relu_gradient(scalar alpha_)
    : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X)
  {
    return eigen::apply(all_relu_derivative(alpha), X);
  }
};

template <typename Matrix>
auto Hyperbolic_tangent(const Matrix& X)
{
  return eigen::apply(hyperbolic_tangent, X);
}

template <typename Matrix>
auto Hyperbolic_tangent_gradient(const Matrix& X)
{
  return eigen::apply(hyperbolic_tangent_derivative, X);
}

template <typename Matrix>
auto Sigmoid(const Matrix& X)
{
  return eigen::apply(sigmoid, X);
}

template <typename Matrix>
auto Sigmoid_gradient(const Matrix& X)
{
  return eigen::apply(sigmoid_derivative, X);
}

struct Srelu
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit Srelu(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
    : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return eigen::apply(srelu(al, tl, ar, tr), X);
  }
};

struct Srelu_gradient
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit Srelu_gradient(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
    : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return eigen::apply(srelu_derivative(al, tl, ar, tr), X);
  }
};

struct Trimmed_relu
{
  scalar epsilon;

  Trimmed_relu(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X)
  {
    return eigen::apply(trimmed_relu(epsilon), X);
  }
};

struct Trimmed_relu_gradient
{
  scalar epsilon;

  Trimmed_relu_gradient(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X)
  {
    return eigen::apply(trimmed_relu_derivative(epsilon), X);
  }
};

/////////////////////////////////////////////////////////////////////////////////////

struct relu_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return Relu(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return Relu_gradient(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return "ReLU()";
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
    return Leaky_relu(alpha)(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return Leaky_relu_gradient(alpha)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("LeakyRelu({})", alpha);
  }
};

struct all_relu_activation
{
  scalar alpha;

  explicit all_relu_activation(scalar alpha_)
    : alpha(alpha_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return All_relu(alpha)(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return All_relu_gradient(alpha)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("AllRelu({})", alpha);
  }
};

struct hyperbolic_tangent_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return Hyperbolic_tangent(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return Hyperbolic_tangent_gradient(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return "HyperbolicTangent()";
  }
};

struct sigmoid_activation
{
  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return Sigmoid(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return Sigmoid_gradient(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return "Sigmoid()";
  }
};

struct trimmed_relu_activation
{
  scalar epsilon;

  explicit trimmed_relu_activation(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return Trimmed_relu(epsilon)(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return Trimmed_relu_gradient(epsilon)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("TReLU({})", epsilon);
  }
};

struct srelu_activation
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit srelu_activation(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
    : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return Srelu(al, tl, ar, tr)(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    return Srelu_gradient(al, tl, ar, tr)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("SReLU({},{},{},{})", al, tl, ar, tr);
  }
};

} // namespace abc

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

  [[nodiscard]] std::string to_string() const
  {
    return "Sigmoid()";
  }
};

struct relu
{
  scalar operator()(scalar x) const
  {
    return std::max(scalar(0), x);
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
    return (x < 0) ? scalar(0) : scalar(1);
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

  [[nodiscard]] std::string to_string() const
  {
    return "ReLU()";
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

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("LeakyRelu({})", alpha);
  }
};

struct trimmed_relu
{
  scalar epsilon;

  explicit trimmed_relu(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  scalar operator()(scalar x) const
  {
    return (x < epsilon) ? scalar(0) : x;
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct trimmed_relu_prime
{
  scalar epsilon;

  explicit trimmed_relu_prime(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  scalar operator()(scalar x) const
  {
    return (x < epsilon) ? scalar(0) : scalar(1);
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct trimmed_relu_activation
{
  scalar epsilon;

  explicit trimmed_relu_activation(scalar epsilon_)
    : epsilon(epsilon_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return trimmed_relu(epsilon)(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return trimmed_relu_prime(epsilon)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("TReLU({})", epsilon);
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

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("AllRelu({})", alpha);
  }
};

struct srelu
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit srelu(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
   : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  scalar operator()(scalar x) const
  {
    if (x <= tl)
    {
      return tl + al * (x - tl);
    }
    else if (x < tr)
    {
      return x;
    }
    else
    {
      return tr + ar * (x - tr);
    }
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct srelu_prime
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit srelu_prime(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
  : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  scalar operator()(scalar x) const
  {
    if (x <= tl)
    {
      return al;
    }
    else if (x < tr)
    {
      return 1;
    }
    else
    {
      return ar;
    }
  }

  template <typename Matrix>
  auto operator()(const Matrix& x) const
  {
    return x.unaryExpr(*this);
  }
};

struct srelu_activation
{
  scalar al;
  scalar tl;
  scalar ar;
  scalar tr;

  explicit srelu_activation(scalar al_ = 0, scalar tl_ = 0, scalar ar_ = 0, scalar tr_ = 1)
  : al(al_), tl(tl_), ar(ar_), tr(tr_)
  {}

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    return srelu(al, tl, ar, tr)(X);
  }

  template <typename Matrix>
  auto prime(const Matrix& X) const
  {
    return srelu_prime(al, tl, ar, tr)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    return fmt::format("SReLU({},{},{},{})", al, tl, ar, tr);
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

  [[nodiscard]] std::string to_string() const
  {
    return "HyperbolicTangent()";
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_ACTIVATION_FUNCTIONS_H
