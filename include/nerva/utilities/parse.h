// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/parse.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_PARSE_H
#define NERVA_UTILITIES_PARSE_H

#include "nerva/utilities/string_utility.h"

namespace nerva::utilities {

inline
std::vector<std::string> parse_arguments(const std::string& text, const std::string& name, unsigned int expected_size)
{
  if (!starts_with(text, name + '('))
  {
    return {};
  }

  auto pos1 = text.find_first_of('(');
  auto pos2 = text.find_last_of(')');
  if (pos1 == std::string::npos || pos2 == std::string::npos)
  {
    throw std::runtime_error("could not parse arguments of string '" + text + "'");
  }
  std::string arguments_text = text.substr(pos1 + 1, pos2 - pos1 - 1);
  auto result = utilities::regex_split(arguments_text, ",");
  if (result.size() != expected_size)
  {
    throw std::runtime_error("the string '" + text + "' has an unexpected number of arguments");
  }
  return result;
}

} // namespace nerva::utilities

#endif // NERVA_UTILITIES_PARSE_H
