// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/loss_functions_rowwise.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/softmax_functions_colwise.h"
#include "nerva/utilities/logger.h"
#include <cmath>
#include <memory>

namespace nerva::rowwise {

struct loss_function
{
  [[nodiscard]] virtual auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar = 0;

  [[nodiscard]] virtual auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix = 0;

  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  virtual ~loss_function() = default;
};

struct squared_error_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return (y - t).squaredNorm() / scalar(2);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return ((Y - T).colwise().squaredNorm() / scalar(2)).sum();
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    return ((Y - T).colwise().squaredNorm() / scalar(2)).sum();
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return Y - T;
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "SquaredErrorLoss()";
  }
};

struct cross_entropy_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    auto log_y = y.unaryExpr([](scalar x) { return std::log(x); });
    return -t.transpose() * log_y;
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    using eigen::hadamard;
    auto log_Y = Y.unaryExpr([](scalar x) { return std::log(x); });
    return (hadamard(-T, log_Y).colwise().sum()).sum();
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    using eigen::elements_sum;
    using eigen::hadamard;
    using eigen::log;

    return elements_sum(hadamard(-T, log(Y)));
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    using eigen::hadamard;
    using eigen::inverse;

    return hadamard(-T, inverse(Y));
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "CrossEntropyLoss()";
  }
};

struct softmax_cross_entropy_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    eigen::vector softmax_y = stable_softmax()(y);
    auto log_softmax_y = softmax_y.unaryExpr([](scalar x) { return std::log(x); });
    return -t.transpose() * log_softmax_y;
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar  // TODO: can the return type become auto?
  {
    using eigen::elements_sum;
    using eigen::hadamard;

    return elements_sum(hadamard(-T, stable_log_softmax()(Y)));
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    using eigen::elements_sum;
    using eigen::hadamard;

    return elements_sum(hadamard(-T, stable_log_softmax()(Y)));
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return stable_softmax()(Y) - T;
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "SoftmaxCrossEntropyLoss()";
  }
};

struct logistic_cross_entropy_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    auto log_1_plus_exp_minus_y = y.unaryExpr([](scalar x) { return std::log1p(std::exp(-x)); });
    return t.transpose() * log_1_plus_exp_minus_y;
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    using eigen::hadamard;
    using eigen::sigmoid;

    auto log_sigmoid_Y = Y.unaryExpr([](scalar x) { return std::log(sigmoid(x)); });
    return (hadamard(-T, log_sigmoid_Y).colwise().sum()).sum();
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    using eigen::elements_sum;
    using eigen::hadamard;
    using eigen::sigmoid;

    auto log_sigmoid_Y = Y.unaryExpr([](scalar x) { return std::log(sigmoid(x)); });
    return elements_sum(hadamard(-T, log_sigmoid_Y));
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    using eigen::hadamard;
    using eigen::sigmoid;

    auto one_minus_sigmoid_Y = Y.unaryExpr([](scalar x) { return scalar(1.0) - sigmoid(x); });
    return hadamard(-T, one_minus_sigmoid_Y);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "LogisticCrossEntropyLoss()";
  }
};

inline
auto parse_loss_function(const std::string& text) -> std::shared_ptr<loss_function>
{
  if (text == "SquaredError")
  {
    return std::make_shared<squared_error_loss>();
  }
  else if (text == "CrossEntropy")
  {
    return std::make_shared<cross_entropy_loss>();
  }
  else if (text == "LogisticCrossEntropy")
  {
    return std::make_shared<logistic_cross_entropy_loss>();
  }
  else if (text == "SoftmaxCrossEntropy")
  {
    return std::make_shared<softmax_cross_entropy_loss>();
  }
  else
  {
    throw std::runtime_error("unknown loss function '" + text + "'");
  }
}

} // namespace nerva::rowwise

#include "nerva/neural_networks/rowwise_colwise.inc"
