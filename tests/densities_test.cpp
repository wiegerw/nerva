// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file densities_test.cpp
/// \brief Tests for computing densities.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/layers.h"
#include <iostream>

TEST_CASE("test_densities")
{
  std::vector<std::pair<long, long>> layer_shapes = {{3072, 1024}, {1024, 512}, {512, 10}};
  float density = 0.05;
  std::vector<float> densities = nerva::compute_sparse_layer_densities(density, layer_shapes);
  std::vector<float> expected_densities = {0.041299715909090914, 0.09292436079545456, 1.0};
  float diff = 0;
  for (std::size_t i = 0; i < densities.size(); i++)
  {
    diff += std::fabs(densities[i] - expected_densities[i]);
  }
  std::cout << "diff = " << diff << std::endl;
  CHECK_LE(diff, 1e-6);
}
