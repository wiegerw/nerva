// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file print_test.cpp
/// \brief Tests for printing floats.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "fmt/format.h"
#include <iostream>

void fmt_test(const std::string& expected, const std::string& result)
{
  if (result != expected)
  {
    std::cout << "  result: '" << result << "'" << std::endl;
    std::cout << "expected: '" << expected << "'" << std::endl;
  }
  CHECK_EQ(expected, result);
}

TEST_CASE("test_fmt")
{
  float f;
  std::string expected;
  std::string result;

  f = 0.12345;
  expected = " 0.1235";
  result = fmt::format("{:7.4f}", f);
  fmt_test(expected, result);

  f = -0.12345;
  expected = "-0.1235";
  result = fmt::format("{:7.4f}", f);
  fmt_test(expected, result);
}
