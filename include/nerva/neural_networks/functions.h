// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/functions.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_FUNCTIONS_H
#define NERVA_NEURAL_NETWORKS_FUNCTIONS_H

#include <cmath>

namespace nerva {

namespace detail {

// Decrements count with one, if possible.
// Returns true if count was decremented.
inline
bool decrement_count(std::size_t& count)
{
  if (count > 0)
  {
    count--;
    return true;
  }
  return false;
}

} // namespace detail

// f(x) = true
struct accept_all
{
  template <typename T>
  bool operator()(T x) const
  {
    return true;
  }
};

// f(x) = (x == 0)
struct accept_zero
{
  template <typename T>
  bool operator()(T x) const
  {
    return x == 0;
  }
};

// f(x) = (x != 0)
struct accept_nonzero
{
  template <typename T>
  bool operator()(T x) const
  {
    return x != 0;
  }
};

// f(x) = (x <= 0)
struct accept_negative
{
  template <typename T>
  bool operator()(T x) const
  {
    return x < 0;
  }
};

// f(x) = (x < 0)
struct accept_strictly_negative
{
  template <typename T>
  bool operator()(T x) const
  {
    return x < 0;
  }
};

// f(x) = (x >= 0)
struct accept_positive
{
  template <typename T>
  bool operator()(T x) const
  {
    return x > 0;
  }
};

// f(x) = (x > 0)
struct accept_strictly_positive
{
  template <typename T>
  bool operator()(T x) const
  {
    return x > 0;
  }
};

// f(x) = (x == value)
template <typename Scalar>
struct accept_value
{
  Scalar value;

  explicit accept_value(Scalar value_)
    : value(value_)
  {}

  bool operator()(Scalar x) const
  {
    return x == value;
  }
};

// f(x) = (|x| <= threshold)
template <typename Scalar>
struct accept_threshold
{
  Scalar threshold;

  explicit accept_threshold(Scalar threshold_)
    : threshold(threshold_)
  {}

  bool operator()(Scalar x) const
  {
    return std::fabs(x) <= threshold;
  }
};

// f(x,y) = |x| < |y|
struct less_magnitude
{
  template <typename T>
  bool operator()(T x, T y) const
  {
    return std::fabs(x) < std::fabs(y);
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_FUNCTIONS_H
