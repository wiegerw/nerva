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
    using eigen::repeat_row;
    using eigen::sum_columns;
    using eigen::exp;
    using eigen::hadamard;
    using eigen::inverse;

    auto m = X.rows();
    auto E = exp(X);
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m));
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
    using eigen::sum_columns;
    using eigen::repeat_row;
    using eigen::exp;
    using eigen::hadamard;

    auto c = X.colwise().maxCoeff().eval();
    auto x_minus_c = X.rowwise() - c;
    auto E = exp(x_minus_c.array());
    auto m = X.rows();
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m));
  }
};

// N.B. Numerically unstable!
struct log_softmax
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    using eigen::repeat_row;
    using eigen::sum_columns;
    using eigen::exp;
    using eigen::log;

    auto N = x.size();
    auto e = log(sum_columns(exp(x)));
    return x.array() - repeat_row(e, N);
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    using eigen::repeat_row;
    using eigen::sum_columns;
    using eigen::exp;
    using eigen::log;

    auto N = X.cols();
    auto E = sum_columns(log(exp(X)));
    return X.array() - repeat_row(E, N);
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
    using eigen::sum_columns;
    using eigen::exp;
    using eigen::log;

    auto c = X.colwise().maxCoeff().eval();
    auto x_minus_c = X.rowwise() - c;
    auto E = exp(x_minus_c);
    return x_minus_c.array().rowwise() - log(sum_columns(E));
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_SOFTMAX_COLWISE_H
