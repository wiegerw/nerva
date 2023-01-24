// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/random.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_RANDOM_H
#define NERVA_NEURAL_NETWORKS_RANDOM_H

#include <random>

namespace nerva {

inline std::mt19937 nerva_rng{std::random_device{}()};

inline
void manual_seed(std::uint_fast32_t seed)
{
  nerva_rng.seed(seed);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_RANDOM_H
