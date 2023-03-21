// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/algorithms.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_ALGORITHMS_H
#define NERVA_UTILITIES_ALGORITHMS_H

#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include "nerva/utilities/random.h"

namespace nerva {

inline
std::vector<std::size_t> reservoir_sample(std::size_t k, std::size_t n, std::mt19937& rng)
{
  std::vector<std::size_t> reservoir(k);
  std::iota(reservoir.begin(), reservoir.end(), 0);

  for (std::size_t i = k; i < n; i++)
  {
    auto j = random_integer<std::size_t>(0, i + 1, rng);
    if (j < k)
    {
      reservoir[j] = i;
    }
  }

  return reservoir;
}

} // namespace nerva

#endif // NERVA_UTILITIES_ALGORITHMS_H
