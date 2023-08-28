// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/softmax_functions_rowwise.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "fmt/format.h"
#include <cmath>
#include <ratio>

namespace nerva::rowwise {

// N.B. Numerically unstable!
struct softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::exp;

    auto E = exp(x);
    return E / E.sum();
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return eigen::softmax_rowwise(X);
  }
};

struct stable_softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::exp;

    // use the log-sum-exp trick to make the computation robust, see also https://en.wikipedia.org/wiki/LogSumExp
    scalar const c = x.maxCoeff();
    auto E = exp((x.array() - c));
    return E / E.sum();
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return eigen::stable_softmax_rowwise(X);
  }
};

// N.B. Numerically unstable!
struct log_softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::exp;
    using eigen::log;

    auto N = x.size();
    auto e = log(columns_sum(exp(x)));
    return x - row_repeat(e, N);
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return eigen::log_softmax_rowwise(X);
  }
};

struct stable_log_softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::exp;

    auto c = x.array().maxCoeff();
    auto E = std::log(exp(x.array() - c).sum());
    return x.array() - c - E;
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return eigen::stable_log_softmax_rowwise(X);
  }
};

} // namespace nerva::rowwise

#include "nerva/neural_networks/rowwise_colwise.inc"
