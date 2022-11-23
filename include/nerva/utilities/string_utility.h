// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/string_utility.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_STRING_UTILITY_H
#define NERVA_UTILITIES_STRING_UTILITY_H

#include <algorithm>
#include <cctype>
#include <locale>
#include <regex>
#include <string>

namespace nerva::utilities {

inline
bool starts_with(const std::string& s, const std::string& prefix)
{
  return s.rfind(prefix, 0) == 0;
}

inline
bool ends_with(const std::string& s, const std::string& postfix)
{
  return (s.length() >= postfix.length()) &&
         (s.compare(s.length() - postfix.length(), postfix.length(), postfix) == 0);
}

// trim from start (in place)
inline void ltrim(std::string& s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
}

// trim from end (in place)
inline void rtrim(std::string& s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string& s)
{
  ltrim(s);
  rtrim(s);
}

// trim from start (copying)
inline std::string ltrim_copy(std::string s)
{
  ltrim(s);
  return s;
}

// trim from end (copying)
inline std::string rtrim_copy(std::string s)
{
  rtrim(s);
  return s;
}

// trim from both ends (copying)
inline std::string trim_copy(std::string s)
{
  trim(s);
  return s;
}

/// \brief Split a string using a regular expression separator.
/// \param text A string
/// \param sep A string
/// \return The splitted string
std::vector<std::string> regex_split(const std::string& text, const std::string& sep);

/// \brief Joins a sequence of strings.
template <typename Container>
std::string string_join(const Container& c, const std::string& separator)
{
  std::ostringstream out;
  for (auto i = c.begin(); i != c.end(); ++i)
  {
    if (i != c.begin())
    {
      out << separator;
    }
    out << *i;
  }
  return out.str();
}

/// \brief Split \c text into paragraphs
std::vector<std::string> split_paragraphs(const std::string& text);

inline
std::vector<std::string> split_lines(const std::string& text)
{
  std::vector<std::string> result;
  std::string::size_type first = 0;
  std::string::size_type last = text.size();

  while (true)
  {
    std::string::size_type next = text.find('\n', first);
    if (next == std::string::npos)
    {
      if (first < last)
      {
        result.emplace_back(text.begin() + first, text.begin() + last);
      }
      break;
    }
    result.emplace_back(text.begin() + first, text.begin() + next);
    first = next + 1;
  }
  return result;
}

/// \brief Match \c line against a regular expression. Throws an exception if the match fails
std::smatch regex_match(const std::string& line, const std::regex& re);

} // namespace nerva::utilities

#endif // NERVA_UTILITIES_STRING_UTILITY_H
