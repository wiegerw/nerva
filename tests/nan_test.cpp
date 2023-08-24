// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nan_test.cpp
/// \brief Tests for NaN values.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include <limits>

TEST_CASE("test_nan")
{
  double x = std::numeric_limits<double>::quiet_NaN();
  CHECK(std::isnan(x));
}
