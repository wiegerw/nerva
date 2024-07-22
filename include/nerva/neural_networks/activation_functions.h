// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/snn/activation_functions.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "fmt/format.h"
#include <cmath>
#include <ratio>
#include <string>

namespace nerva::eigen {

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

  explicit Leaky_relu(scalar alpha_)
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

  explicit Leaky_relu_gradient(scalar alpha_)
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

  explicit All_relu(scalar alpha_)
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

  explicit All_relu_gradient(scalar alpha_)
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

  explicit Trimmed_relu(scalar epsilon_)
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

  explicit Trimmed_relu_gradient(scalar epsilon_)
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
  // N.B. The parameters are stored in matrices so that an optimizer can be used
  eigen::matrix x;  // contains al, tl, ar, tr
  eigen::matrix Dx; // contains Dal, Dtl, Dar, Dtr

  explicit srelu_activation(scalar al = 0, scalar tl = 0, scalar ar = 0, scalar tr = 1)
  {
    x = eigen::matrix{{al, tl, ar, tr}};
    Dx = eigen::matrix{{0, 0, 0, 0}};
  }

  template <typename Matrix>
  auto operator()(const Matrix& X) const
  {
    auto al = x(0, 0);
    auto tl = x(0, 1);
    auto ar = x(0, 2);
    auto tr = x(0, 3);
    return Srelu(al, tl, ar, tr)(X);
  }

  template <typename Matrix>
  auto gradient(const Matrix& X) const
  {
    auto al = x(0, 0);
    auto tl = x(0, 1);
    auto ar = x(0, 2);
    auto tr = x(0, 3);
    return Srelu_gradient(al, tl, ar, tr)(X);
  }

  [[nodiscard]] std::string to_string() const
  {
    auto al = x(0, 0);
    auto tl = x(0, 1);
    auto ar = x(0, 2);
    auto tr = x(0, 3);
    return fmt::format("SReLU(al={},tl={},ar={},tr={})", al, tl, ar, tr);
  }
};

} // namespace nerva::eigen

