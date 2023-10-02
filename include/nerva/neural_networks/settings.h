// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/settings.h
/// \brief add your file description here.

#pragma once

#include <stdexcept>

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

inline computation NervaComputation = computation::eigen;

inline
void set_nerva_computation(const std::string& text)
{
  if (text == "eigen")
  {
    NervaComputation = computation::eigen;
  }
  else if (text == "mkl")
  {
    NervaComputation = computation::mkl;
  }
  else if (text == "blas")
  {
    NervaComputation = computation::blas;
  }
  else
  {
    throw std::runtime_error("unknown computation " + text);
  }
}

} // namespace nerva
