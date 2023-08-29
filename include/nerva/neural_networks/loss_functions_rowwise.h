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
#include "nerva/neural_networks/loss_functions.h"
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
    return nerva::eigen::squared_error_loss_rowwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return nerva::eigen::Squared_error_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    return nerva::eigen::Squared_error_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return nerva::eigen::Squared_error_loss_rowwise_gradient(Y, T);
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
    return nerva::eigen::cross_entropy_loss_rowwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return nerva::eigen::Cross_entropy_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    return nerva::eigen::Cross_entropy_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return nerva::eigen::Cross_entropy_loss_rowwise_gradient(Y, T);
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
    return nerva::eigen::softmax_cross_entropy_loss_rowwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return nerva::eigen::Softmax_cross_entropy_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    return nerva::eigen::Softmax_cross_entropy_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return nerva::eigen::Softmax_cross_entropy_loss_rowwise_gradient(Y, T);
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
    return nerva::eigen::logistic_cross_entropy_loss_rowwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return nerva::eigen::Logistic_cross_entropy_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    return nerva::eigen::Logistic_cross_entropy_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return nerva::eigen::Logistic_cross_entropy_loss_rowwise_gradient(Y, T);
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "LogisticCrossEntropyLoss()";
  }
};

struct negative_log_likelihood_loss: public loss_function
{
  template <typename Target>
  auto operator()(const eigen::vector& y, const Target& t) const -> scalar
  {
    return nerva::eigen::negative_log_likelihood_loss_rowwise(y, t);
  }

  template <typename Target>
  auto operator()(const eigen::matrix& Y, const Target& T) const -> scalar
  {
    return nerva::eigen::Negative_log_likelihood_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto value(const eigen::matrix& Y, const eigen::matrix& T) const -> scalar override
  {
    return nerva::eigen::Negative_log_likelihood_loss_rowwise(Y, T);
  }

  [[nodiscard]] auto gradient(const eigen::matrix& Y, const eigen::matrix& T) const -> eigen::matrix override
  {
    return nerva::eigen::Negative_log_likelihood_loss_rowwise_gradient(Y, T);
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
  else if (text == "NegativeLogLikelyhood")
  {
    return std::make_shared<negative_log_likelihood_loss>();
  }
  else
  {
    throw std::runtime_error("unknown loss function '" + text + "'");
  }
}

} // namespace nerva::rowwise

#include "nerva/neural_networks/rowwise_colwise.inc"
