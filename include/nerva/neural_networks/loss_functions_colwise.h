// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/loss_functions_colwise.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_COLWISE_H
#define NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_COLWISE_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/softmax_colwise.h"
#include "nerva/utilities/logger.h"
#include <cmath>
#include <memory>

namespace nerva {

struct loss_function
{
  [[nodiscard]] virtual scalar value(const eigen::matrix& Y, const eigen::matrix& T) const = 0;

  [[nodiscard]] virtual eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const = 0;

  [[nodiscard]] virtual std::string to_string() const = 0;

  virtual ~loss_function() = default;
};

struct squared_error_loss: public loss_function
{
  template <typename Target>
  scalar operator()(const eigen::vector& y, const Target& t) const
  {
    return (y - t).squaredNorm() / scalar(2);
  }

  template <typename Target>
  scalar operator()(const eigen::matrix& Y, const Target& T) const
  {
    return ((Y - T).colwise().squaredNorm() / scalar(2)).sum();
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return ((Y - T).colwise().squaredNorm() / scalar(2)).sum();
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Y - T;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "SquaredErrorLoss()";
  }
};

struct cross_entropy_loss: public loss_function
{
  template <typename Target>
  scalar operator()(const eigen::vector& y, const Target& t) const
  {
    auto log_y = y.unaryExpr([](scalar x) { return std::log(x); });
    return -t.transpose() * log_y;
  }

  template <typename Target>
  scalar operator()(const eigen::matrix& Y, const Target& T) const
  {
    using eigen::hadamard;
    auto log_Y = Y.unaryExpr([](scalar x) { return std::log(x); });
    return (hadamard(-T, log_Y).colwise().sum()).sum();
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    using eigen::elements_sum;
    using eigen::hadamard;
    using eigen::log;

    return elements_sum(hadamard(-T, log(Y)));
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    using eigen::hadamard;
    using eigen::inverse;

    return hadamard(-T, inverse(Y));
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "CrossEntropyLoss()";
  }
};

struct softmax_cross_entropy_loss: public loss_function
{
  template <typename Target>
  scalar operator()(const eigen::vector& y, const Target& t) const
  {
    eigen::vector softmax_y = stable_softmax()(y);
    auto log_softmax_y = softmax_y.unaryExpr([](scalar x) { return std::log(x); });
    return -t.transpose() * log_softmax_y;
  }

  template <typename Target>
  scalar operator()(const eigen::matrix& Y, const Target& T) const  // TODO: can the return type become auto?
  {
    using eigen::elements_sum;
    using eigen::hadamard;

    return elements_sum(hadamard(-T, stable_log_softmax()(Y)));
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    using eigen::elements_sum;
    using eigen::hadamard;

    return elements_sum(hadamard(-T, stable_log_softmax()(Y)));
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return stable_softmax()(Y) - T;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "SoftmaxCrossEntropyLoss()";
  }
};

struct logistic_cross_entropy_loss: public loss_function
{
  template <typename Target>
  scalar operator()(const eigen::vector& y, const Target& t) const
  {
    auto log_1_plus_exp_minus_y = y.unaryExpr([](scalar x) { return std::log1p(std::exp(-x)); });
    return t.transpose() * log_1_plus_exp_minus_y;
  }

  template <typename Target>
  scalar operator()(const eigen::matrix& Y, const Target& T) const
  {
    using eigen::hadamard;
    using eigen::sigmoid;

    auto log_sigmoid_Y = Y.unaryExpr([](scalar x) { return std::log(sigmoid(x)); });
    return (hadamard(-T, log_sigmoid_Y).colwise().sum()).sum();
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    using eigen::elements_sum;
    using eigen::hadamard;
    using eigen::sigmoid;

    auto log_sigmoid_Y = Y.unaryExpr([](scalar x) { return std::log(sigmoid(x)); });
    return elements_sum(hadamard(-T, log_sigmoid_Y));
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    using eigen::hadamard;
    using eigen::sigmoid;

    auto one_minus_sigmoid_Y = Y.unaryExpr([](scalar x) { return scalar(1.0) - sigmoid(x); });
    return hadamard(-T, one_minus_sigmoid_Y);
  }

  [[nodiscard]] std::string to_string() const override
  {
    return "LogisticCrossEntropyLoss()";
  }
};

inline
std::shared_ptr<loss_function> parse_loss_function(const std::string& text)
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

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_COLWISE_H
