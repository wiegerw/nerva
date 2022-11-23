// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file src/utilities.cpp
/// \brief add your file description here.

#include <regex>
#include "nerva/utilities/string_utility.h"

namespace nerva::utilities {

std::vector<std::string> regex_split(const std::string& text, const std::string& sep)
{
  std::vector<std::string> result;
  std::regex re(sep);
  std::sregex_token_iterator i(text.begin(), text.end(), re, -1);
  std::sregex_token_iterator end;
  while (i != end)
  {
    std::string word = i->str();
    trim(word);
    if (!word.empty())
    {
      result.push_back(word);
    }
    ++i;
  }
  return result;
}

std::smatch regex_match(const std::string& line, const std::regex& re)
{
  std::smatch m;
  bool result = std::regex_match(line, m, re);
  if (!result)
  {
    throw std::runtime_error("Could not parse line '" + line + "'");
  }
  return m;
}

std::vector<std::string> split_paragraphs(const std::string& text)
{
  std::vector<std::string> result;

  // find multiple line endings
  std::regex sep(R"(\n\s*\n)");

  // the -1 below directs the token iterator to display the parts of
  // the string that did NOT match the regular expression.
  std::sregex_token_iterator cur(text.begin(), text.end(), sep, -1);
  std::sregex_token_iterator end;

  for (; cur != end; ++cur)
  {
    std::string paragraph = *cur;
    trim(paragraph);
    if (!paragraph.empty())
    {
      result.push_back(paragraph);
    }
  }
  return result;
}

} // namespace nerva::utilities