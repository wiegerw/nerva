// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/loss_functions.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_H
#define NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/activation_functions.h"
#include "nerva/utilities/logger.h"
#include <cmath>
#include <memory>

namespace nerva {

struct loss_function
{
  [[nodiscard]] virtual scalar value(const eigen::matrix& Y, const eigen::matrix& T) const = 0;

  [[nodiscard]] virtual eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const = 0;

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

  template <typename Target>
  eigen::vector gradient_vector(const eigen::vector& y, const Target& t) const
  {
    return y - t;
  }

  template <typename Target>
  auto gradient_matrix(const eigen::matrix& Y, const Target& T) const
  {
    return Y - T;
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return ((Y - T).colwise().squaredNorm() / scalar(2)).sum();
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return Y - T;
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
    auto log_Y = Y.unaryExpr([](scalar x) { return std::log(x); });
    return (-T.cwiseProduct(log_Y).colwise().sum()).sum();
  }

  template <typename Target>
  eigen::vector gradient_vector(const eigen::vector& y, const Target& t) const
  {
    auto one_div_y = y.unaryExpr([](scalar x) { return scalar(1.0) / x; });
    return -t.cwiseProduct(one_div_y);
  }

  template <typename Target>
  auto gradient_matrix(const eigen::matrix& Y, const Target& T) const
  {
    auto one_div_Y = Y.unaryExpr([](scalar x) { return scalar(1.0) / x; });
    return -T.cwiseProduct(one_div_Y);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    auto log_Y = Y.unaryExpr([](scalar x) { return std::log(x); });
    return (-T.cwiseProduct(log_Y).colwise().sum()).sum();
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    auto one_div_Y = Y.unaryExpr([](scalar x) { return scalar(1.0) / x; });
    return -T.cwiseProduct(one_div_Y);
  }
};

struct softmax_cross_entropy_loss: public loss_function
{
  template <typename Target>
  scalar operator()(const eigen::vector& y, const Target& t) const
  {
    eigen::vector softmax_y = softmax()(y);
    auto log_softmax_y = softmax_y.unaryExpr([](scalar x) { return std::log(x); });
    return -t.transpose() * log_softmax_y;
  }

  template <typename Target>
  scalar operator()(const eigen::matrix& Y, const Target& T) const  // TODO: can the return type become auto?
  {
    eigen::matrix log_softmax_Y = softmax().log(Y);
    return (-T.cwiseProduct(log_softmax_Y).colwise().sum()).sum();
  }

  template <typename Target>
  eigen::vector gradient_vector(const eigen::vector& y, const Target& t) const
  {
    return softmax()(y) - t;
  }

  template <typename Target>
  eigen::matrix gradient_matrix(const eigen::matrix& Y, const Target& T) const  // TODO: can the return type become auto?
  {
    return softmax()(Y) - T;
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    eigen::matrix log_softmax_Y = softmax().log(Y);
    return (-T.cwiseProduct(log_softmax_Y).colwise().sum()).sum();
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    return softmax()(Y) - T;
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
    auto log_sigmoid_Y = Y.unaryExpr([](scalar x) { return std::log(sigmoid()(x)); });
    return (-T.cwiseProduct(log_sigmoid_Y).colwise().sum()).sum();
  }

  template <typename Target>
  eigen::vector gradient_vector(const eigen::vector& y, const Target& t) const
  {
    auto one_minus_sigmoid_y = y.unaryExpr([](scalar x) { return scalar(1.0) - sigmoid()(x); });
    return -t.cwiseProduct(one_minus_sigmoid_y);
  }

  template <typename Target>
  auto gradient_matrix(const eigen::matrix& Y, const Target& T) const
  {
    auto one_minus_sigmoid_Y = Y.unaryExpr([](scalar x) { return scalar(1.0) - sigmoid()(x); });
    return -T.cwiseProduct(one_minus_sigmoid_Y);
  }

  [[nodiscard]] scalar value(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    auto log_sigmoid_Y = Y.unaryExpr([](scalar x) { return std::log(sigmoid()(x)); });
    return (-T.cwiseProduct(log_sigmoid_Y).colwise().sum()).sum();
  }

  [[nodiscard]] eigen::matrix gradient(const eigen::matrix& Y, const eigen::matrix& T) const override
  {
    auto one_minus_sigmoid_Y = Y.unaryExpr([](scalar x) { return scalar(1.0) - sigmoid()(x); });
    return -T.cwiseProduct(one_minus_sigmoid_Y);
  }
};

inline
std::shared_ptr<loss_function> parse_loss_function(const std::string& text)
{
  if (text == "squared-error")
  {
    return std::make_shared<squared_error_loss>();
  }
  else if (text == "cross-entropy")
  {
    return std::make_shared<cross_entropy_loss>();
  }
  else if (text == "logistic-cross-entropy")
  {
    return std::make_shared<logistic_cross_entropy_loss>();
  }
  else if (text == "softmax-cross-entropy")
  {
    return std::make_shared<softmax_cross_entropy_loss>();
  }
  else
  {
    throw std::runtime_error("Unknown loss function '" + text + "'");
  }
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_H
