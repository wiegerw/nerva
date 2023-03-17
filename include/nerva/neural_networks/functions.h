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

// f(x) = (x < 0)
struct accept_negative
{
  template <typename T>
  bool operator()(T x) const
  {
    return x < 0;
  }
};

// f(x) = (x > 0)
struct accept_positive
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

// f(x) = 0 < |x| <= threshold
//
// At most \a threshold_count elements with |x| == threshold are accepted.
template <typename Scalar>
struct accept_nonzero_threshold
{
  Scalar threshold;
  mutable std::size_t threshold_count;

  accept_nonzero_threshold(Scalar threshold_, std::size_t threshold_count_)
   : threshold(threshold_), threshold_count(threshold_count_)
  {}

  template <typename T>
  bool operator()(T x) const
  {
    auto x_ = std::fabs(x);
    if (0 != x_ && x_ <= threshold)
    {
      if (std::fabs(x) == threshold)
      {
        if (threshold_count > 0)
        {
          threshold_count--;
          return true;
        }
        return false;
      }
      return true;
    }
    return false;
  }
};


// Accepts values x with `threshold_negative <= x < 0` or `0 < x <= threshold_positive`.
// At most num_copies_negative elements with x == threshold_negative are accepted.
// At most num_copies_positive elements with x == threshold_positive are accepted.
template <typename Scalar>
struct accept_nonzero_threshold_positive_negative
{
  Scalar threshold_negative;
  Scalar threshold_positive;
  mutable std::size_t threshold_negative_count;
  mutable std::size_t threshold_positive_count;

  accept_nonzero_threshold_positive_negative(Scalar threshold_negative_,
                                             Scalar threshold_positive_,
                                             std::size_t threshold_negative_count_,
                                             std::size_t threshold_positive_count_)
    : threshold_negative(threshold_negative_),
      threshold_positive(threshold_positive_),
      threshold_negative_count(threshold_negative_count_),
      threshold_positive_count(threshold_positive_count_)
  {}

  template <typename T>
  bool operator()(T x) const
  {
    if (x < 0 && threshold_negative <= x)
    {
      if (x == threshold_negative)
      {
        return threshold_negative_count-- > 0;
      }
      return true;
    }
    else if (x > 0 && x <= threshold_positive)
    {
      if (x == threshold_positive)
      {
        return threshold_positive_count-- > 0;
      }
      return true;
    }
    return false;
  }
};

// f(x,y) = |x| < |y|
struct compare_less_absolute
{
  template <typename T>
  bool operator()(T x, T y) const
  {
    return std::fabs(x) < std::fabs(y);
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_FUNCTIONS_H
