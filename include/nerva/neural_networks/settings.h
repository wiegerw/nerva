// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/settings.h
/// \brief add your file description here.

#pragma once

namespace nerva {

#ifdef NERVA_USE_DOUBLE
using scalar = double;
#else
using scalar = float;
#endif

enum class computation
{
  eigen,
  mkl,
  blas
};

#ifdef NERVA_USE_MKL
constexpr computation NervaComputation = computation::mkl;
#elif defined NERVA_USE_BLAS
constexpr computation NervaComputation = computation::blas;
#else
constexpr computation NervaComputation = computation::eigen;
#endif

} // namespace nerva
