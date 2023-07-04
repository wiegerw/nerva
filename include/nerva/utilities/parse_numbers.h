// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/parse_numbers.h
/// \brief add your file description here.

#ifndef NERVA_PARSE_NUMBERS_H
#define NERVA_PARSE_NUMBERS_H

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "nerva/utilities/logger.h"
#include "nerva/neural_networks/scalar.h"
#include "nerva/utilities/string_utility.h"

namespace nerva {

namespace detail {

// Reads the next integer from the range [first, last), and the spaces behind it
// Returns the position in the range after the next integer
// Precondition: (first != last) && !std::isspace(*first)
template <typename Number = std::size_t, typename Iterator>
Iterator parse_next_natural_number(Iterator first, Iterator last, Number& result)
{
  assert((first != last) && !std::isspace(*first));

  Iterator i = first;
  result = 0;

  for (;;)
  {
    if (*i < '0' || *i > '9')
    {
      throw std::runtime_error("could not read an integer from " + std::string(first, last));
    }
    result *= 10;
    result += *i - '0';
    ++i;
    if (i == last)
    {
      break;
    }
    if (std::isspace(*i))
    {
      ++i;
      while (i != last && std::isspace(*i))
      {
        ++i;
      }
      break;
    }
  }
  return i;
}

} // namespace detail

/// \brief Parses a natural number from a string
template <typename Number = std::size_t, typename Iterator>
Number parse_natural_number(Iterator first, Iterator last)
{
  // skip leading spaces
  while (first != last && std::isspace(*first))
  {
    ++first;
  }

  if (first == last)
  {
    throw std::runtime_error("could not read an integer from " + std::string(first, last));
  }

  Number value;
  first = detail::parse_next_natural_number(first, last, value);

  if (first != last)
  {
    throw std::runtime_error("could not read an integer from " + std::string(first, last));
  }

  return value;
}

template <typename Number = std::size_t>
Number parse_natural_number(const std::string& text)
{
  return parse_natural_number(text.begin(), text.end());
}

/// \brief Parses a sequence of natural numbers (separated by spaces) from a string
template <typename Number = std::size_t, typename Iterator>
std::vector<Number> parse_natural_number_sequence(Iterator first, Iterator last)
{
  std::vector<Number> result;

  // skip leading spaces
  while (first != last && std::isspace(*first))
  {
    ++first;
  }

  while (first != last)
  {
    Number value;
    first = detail::parse_next_natural_number(first, last, value);
    result.push_back(value);
  }

  return result;
}

template <typename Number = std::size_t>
std::vector<Number> parse_natural_number_sequence(const std::string& text)
{
  return parse_natural_number_sequence(text.begin(), text.end());
}

inline
double parse_double(const std::string& text)
{
  // Use stold instead of stod to avoid out_of_range errors
  return std::stold(text, nullptr);
}

inline
scalar parse_scalar(const std::string& text)
{
  if constexpr (std::is_same<scalar, double>::value)
  {
    return std::stold(text, nullptr);
  }
  else
  {
    return std::stof(text, nullptr);
  }
}

inline
std::vector<double> parse_double_sequence(const std::string& text)
{
  std::vector<double> result;
  const char* p = text.c_str();
  char* end;
  // Use strtold instead of strtod to avoid out_of_range errors
  for (double f = std::strtold(p, &end); p != end; f = std::strtold(p, &end))
  {
    p = end;
    if (errno == ERANGE)
    {
      NERVA_LOG(log::debug) << "WARNING: parse_double_sequence got a range error in ";
      errno = 0;
    }
    result.push_back(f);
  }
  return result;
}

inline
uint32_t parse_binary_number(const std::string& text)
{
  return std::stoll(text, nullptr, 2);
}

template <typename Iterator>
Iterator skip_spaces(Iterator first, Iterator last)
{
  while (first != last && std::isspace(*first))
  {
    ++first;
  }
  return first;
}

template <typename Iterator>
Iterator skip_string(Iterator first, Iterator last, const std::string& s)
{
  first = first + std::string(s).size();
  first = skip_spaces(first, last);
  return first;
}

// Reads an integer from the range [first, last).
// - Skips spaces at the front.
// - If no digits were found, result is unchanged.
// Returns the position in the range after the integer
template <typename Iterator>
Iterator parse_integer(Iterator first, Iterator last, unsigned int& result)
{
  auto is_digit = [&](Iterator i)
  {
    return '0' <= *i && *i <= '9';
  };

  Iterator i = skip_spaces(first, last);
  if (i == last || !is_digit(i))
  {
    return i;
  }

  result = *i - '0';
  ++i;

  while (i != last && is_digit(i))
  {
    result *= 10;
    result += *i - '0';
    ++i;
  }
  return i;
}

template <typename Iterator>
Iterator parse_double(Iterator first, double& result)
{
  const char* begin = &(*first);
  char* end;

  result = std::strtod(begin, &end);
  if (errno == ERANGE)
  {
    std::cout << "range error, got ";
    errno = 0;
  }

  return first + (end - begin);
}

template <>
const char* parse_double(const char* first, double& result)
{
  char* end;

  result = std::strtod(first, &end);
  if (errno == ERANGE)
  {
    std::cout << "range error, got ";
    errno = 0;
  }

  return const_cast<const char*>(end);
}

template <typename Number = std::size_t>
std::vector<Number> parse_comma_separated_numbers(const std::string& text)
{
  std::vector<Number> result;
  for (const std::string& word: utilities::regex_split(text, ","))
  {
    result.push_back(parse_natural_number<Number>(word));
  }
  return result;
}

} // namespace nerva

#endif // NERVA_PARSE_NUMBERS_H
