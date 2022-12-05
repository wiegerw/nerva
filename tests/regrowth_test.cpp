// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file regrowth_test.cpp
/// \brief Tests for regrowing of a sparse weight matrix.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/mkl_matrix.h"
#include "nerva/neural_networks/regrowth.h"

using namespace nerva;

TEST_CASE("test1")
{
  const eigen::matrix A {
    {1, -1, 0, -2, 0},
    {-2, 5, 0, 0, 0},
    {0, 0, 4, 6, 4},
    {-4, 0, 3, 7, 0},
    {0, 8, 0, 0, -5}
  };

  eigen::matrix A_pruned_expected {
    {0, 0, 0, 0, 0},
    {0, 5, 0, 0, 0},
    {0, 0, 4, 6, 4},
    {-4, 0, 0, 7, 0},
    {0, 8, 0, 0, -5}
  };

  eigen::print_matrix("A", A);

  auto f = []() { return scalar(10); };
  long k = 5;
  std::mt19937 rng{std::random_device{}()};

  auto threshold = find_k_smallest_value(A, k);
  CHECK_EQ(3, threshold);

  auto A_pruned = A;
  long prune_count = prune(A_pruned, threshold);
  eigen::print_matrix("A_pruned", A_pruned);
  CHECK_EQ(A_pruned_expected, A_pruned);
  CHECK_EQ(5, prune_count);

  auto A_grow = A_pruned;
  grow(A_grow, f, prune_count, rng);
  eigen::print_matrix("A_grow", A_grow);

  long m = A.rows();
  long n = A.cols();

  // Check if the large weights are unchanged
  for (long i = 0; i < m; i++)
  {
    for (long j = 0; j < n; j++)
    {
      if (std::fabs(A(i, j)) > threshold)
      {
        CHECK_EQ(A(i, j), A_grow(i, j));
      }
    }
  }
  CHECK_EQ((A_grow.array() == 10).count(), prune_count);
  CHECK_EQ((A_grow.array() == 0).count(), (A.array() == 0).count());

  auto B = A;
  regrow(B, f, k, rng);
  CHECK_EQ((B.array() == 10).count(), prune_count);
  CHECK_EQ((B.array() == 0).count(), (A.array() == 0).count());
}
