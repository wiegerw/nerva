// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/random.h
/// \brief add your file description here.

#ifndef NERVA_RANDOM_H
#define NERVA_RANDOM_H

#include <random>

namespace nerva {

/// \brief Returns a random floating point value in the range [low; high).
/// Random numbers are generated using the random number generator g.
template <typename Real, typename URBG>
Real random_real(Real low, Real high, URBG&& g)
{
  std::uniform_real_distribution<Real> dist(low, high);
  return dist(g);
}

/// \brief Returns a random floating point value in the range [low; high).
template <class Real>
Real random_real(Real low, Real high)
{
  return random_real(low, high, std::mt19937{std::random_device{}()});
}

/// \brief Returns a random integer value in the range [low; high).
/// Random numbers are generated using the random number generator g.
template <class Integer, class URBG>
Integer random_integer(Integer low, Integer high, URBG&& g)
{
  std::uniform_int_distribution<Integer> dist(low, high);
  return dist(g);
}

/// \brief Returns a random integer value in the range [low; high).
template <class Integer>
Integer random_integer(Integer low, Integer high)
{
  return random_integer(low, high, std::mt19937{std::random_device{}()});
}

/// \brief Returns a random boolean value.
/// Random numbers are generated using the random number generator g.
template <class URBG>
bool random_bool(URBG&& g)
{
  return random_integer(0, 1, g) == 0;
}

/// \brief Returns a random boolean value.
inline
bool random_bool()
{
  return random_bool(std::mt19937{std::random_device{}()});
}

/// \brief Selects n elements from the sequence [first; last) (without replacement) such that each possible
/// sample has equal probability of appearance, and writes those selected elements into the output iterator out.
/// Random numbers are generated using the random number generator g.
/// \return Returns a copy of out after the last sample that was output, that is, end of the sample range.
template <class PopulationIterator, class SampleIterator, class Distance, class URBG>
SampleIterator sample_with_replacement(PopulationIterator first, PopulationIterator last, SampleIterator out, Distance n, URBG&& g)
{
  const std::size_t m = std::distance(first, last);
  std::uniform_int_distribution<std::size_t> dist(0, m-1);
  for (std::size_t i = 0; i < n; i++)
  {
    auto it = first + dist(g);
    *(out++) = *it;
  }
  return out;
}

} // namespace nerva

#endif // NERVA_RANDOM_H
