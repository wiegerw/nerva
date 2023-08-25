// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/softmax_rowwise.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "fmt/format.h"
#include <cmath>
#include <ratio>

namespace nerva {

// N.B. Numerically unstable!
struct softmax_rowwise
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    // TODO
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    // TODO
  }
};

struct stable_softmax_rowwise
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    // TODO
  }

  // see also https://gist.github.com/WilliamTambellini/8294f211800e16791d47f3cf59472a49
  eigen::matrix operator()(const eigen::matrix& X) const
  {
    // TODO
  }
};

struct stable_log_softmax_rowwise
{
  [[nodiscard]] eigen::vector value(const eigen::vector& x) const
  {
    // TODO
  }

  eigen::matrix operator()(const eigen::matrix& X) const
  {
    // TODO
  }
};

} // namespace nerva

