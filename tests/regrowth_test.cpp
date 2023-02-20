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
#include <algorithm>

using namespace nerva;

TEST_CASE("test_nth_element")
{
  std::vector<int> v;
  std::vector<int>::iterator w;

  v = {-2, -4, -1, -2, -5};
  w = v.begin() + 1;
  std::nth_element(v.begin(), w, v.end(), std::greater<>());
  CHECK_EQ(-2, *w);

  v = {-2, -4, -1, -2, -5};
  w = v.begin() + 2;
  std::nth_element(v.begin(), w, v.end(), std::greater<>());
  CHECK_EQ(-2, *w);

  v = {-2, -4, -1, -2, -5};
  w = v.begin() + 3;
  std::nth_element(v.begin(), w, v.end(), std::greater<>());
  CHECK_EQ(-4, *w);

  v = {1, 5, 8, 4, 3, 6, 7, 4};
  w = v.begin() + 2;
  std::nth_element(v.begin(), w, v.end());
  CHECK_EQ(4, *w);

  v = {1, 5, 8, 4, 3, 6, 7, 4};
  w = v.begin() + 3;
  std::nth_element(v.begin(), w, v.end());
  CHECK_EQ(4, *w);

  v = {1, 5, 8, 4, 3, 6, 7, 4};
  w = v.begin() + 4;
  std::nth_element(v.begin(), w, v.end());
  CHECK_EQ(5, *w);
}

TEST_CASE("test1")
{
  std::cout << "--- test1 ---\n";

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

  std::mt19937 rng{std::random_device{}()};
  auto init = std::make_shared<ten_weight_initializer>(rng);

  long k = 4; // consider the 5 smallest nonzero elements
  scalar threshold;
  unsigned int num_copies;
  std::tie(threshold, num_copies) = nth_element(A, k, accept_nonzero(), compare_less_absolute());
  CHECK_EQ(3, threshold);
  CHECK_EQ(1, num_copies);

  auto A_pruned = A;
  auto accept = [threshold](scalar x) { return x != 0 && std::fabs(x) <= threshold; };
  long prune_count = prune(A_pruned, accept);
  eigen::print_matrix("A_pruned", A_pruned);
  CHECK_EQ(A_pruned_expected, A_pruned);
  CHECK_EQ(5, prune_count);

  auto A_grow = A_pruned;
  grow(A_grow, init, prune_count, rng);
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

  long zero_count = (A.array() == 0).count();
  long nonzero_count = (A.array() != 0).count();
  for (long count = 1; count <= nonzero_count; count++)
  {
    if (count > zero_count)
    {
      continue;
    }
    std::cout << "=== regrow_threshold count = " << count << " ===" << std::endl;
    auto B = A;
    eigen::print_matrix("A", A);
    regrow_threshold(B, init, count, rng);
    eigen::print_matrix("B", B);
    CHECK_EQ((B.array() == 10).count(), count);
    CHECK_EQ((B.array() == 0).count(), (A.array() == 0).count());
  }
}

TEST_CASE("test2")
{
  std::cout << "--- test2 ---\n";

  const eigen::matrix A {
    { 1, -1, 0, -2,  0},
    {-2,  5, 0,  0,  0},
    { 0,  0, 4,  6,  4},
    {-4,  0, 3,  7,  0},
    { 0,  8, 0,  0, -5}
  };

  eigen::matrix A_pruned_expected {
    {0, 0, 0, 0, 0},
    {0, 5, 0, 0, 0},
    {0, 0, 0, 6, 0},
    {-4, 0, 0, 7, 0},
    {0, 8, 0, 0, -5}
  };

  eigen::print_matrix("A", A);

  auto A_pruned = A;
  float negative_threshold = -2;
  float positive_threshold = 4;
  auto accept = [negative_threshold, positive_threshold](scalar x)
  {
    return (negative_threshold <= x && x < 0) || (0 < x && x <= positive_threshold);
  };
  long prune_count = prune(A_pruned, accept);
  eigen::print_matrix("A_pruned_expected", A_pruned_expected);
  CHECK_EQ(A_pruned_expected, A_pruned);
  CHECK_EQ(7, prune_count);

  long max_negative_count = (A.array() < 0).count();
  long max_positive_count = (A.array() > 0).count();
  long zero_count = (A.array() == 0).count();
  for (long negative_count = 1; negative_count <= max_negative_count; negative_count++)
  {
    for (long positive_count = 1; positive_count <= max_positive_count; positive_count++)
    {
      if (negative_count + positive_count > zero_count)
      {
        continue;
      }
      std::cout << "=== regrow_interval negative_count = " << negative_count << " positive_count = " << positive_count << " ===" << std::endl;
      eigen::print_matrix("A", A);
      auto B = A;
      std::mt19937 rng{std::random_device{}()};
      auto init = std::make_shared<ten_weight_initializer>(rng);
      regrow_interval(B, init, negative_count, positive_count, rng);
      eigen::print_matrix("B", B);
      CHECK_EQ((B.array() == 10).count(), negative_count + positive_count);
      CHECK_EQ((B.array() == 0).count(), (A.array() == 0).count());
    }
  }
}
