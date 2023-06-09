// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/softmax_colwise.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_SOFTMAX_COLWISE_H
#define NERVA_NEURAL_NETWORKS_SOFTMAX_COLWISE_H

#include "nerva/neural_networks/eigen.h"
#include "fmt/format.h"
#include <cmath>
#include <ratio>

namespace nerva {

// N.B. Numerically unstable!
struct softmax
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    using eigen::exp;

    auto E = exp(x);
    return E / E.sum();
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::exp;
    using eigen::hadamard;
    using eigen::inverse;

    auto D = X.rows();
    auto E = exp(X);
    return hadamard(E, row_repeat(inverse(columns_sum(E)), D));
  }
};

struct stable_softmax
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    using eigen::exp;

    // use the log-sum-exp trick to make the computation robust, see also https://en.wikipedia.org/wiki/LogSumExp
    scalar c = x.maxCoeff();
    auto E = exp((x.array() - c));
    return E / E.sum();
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    using eigen::columns_sum;
    using eigen::columns_max;
    using eigen::exp;
    using eigen::inverse;
    using eigen::row_repeat;
    using eigen::hadamard;

    auto D = X.rows();
    auto Y = X - row_repeat(columns_max(X), D);
    auto E = exp(Y);
    return hadamard(E, row_repeat(inverse(columns_sum(E)), D));
  }
};

// N.B. Numerically unstable!
struct log_softmax
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::exp;
    using eigen::log;

    auto N = x.size();
    auto e = log(columns_sum(exp(x)));
    return x - row_repeat(e, N);
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::exp;
    using eigen::log;

    auto N = X.cols();
    auto E = columns_sum(log(exp(X)));
    return X - row_repeat(E, N);
  }
};

struct stable_log_softmax
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    using eigen::exp;

    auto c = x.array().maxCoeff();
    auto E = std::log(exp(x.array() - c).sum());
    return x.array() - c - E;
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    using eigen::columns_sum;
    using eigen::columns_max;
    using eigen::exp;
    using eigen::log;
    using eigen::row_repeat;

    auto D = X.rows();
    auto Y = X - row_repeat(columns_max(X), D);
    return Y - row_repeat(log(columns_sum(exp(Y))), D);
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_SOFTMAX_COLWISE_H
