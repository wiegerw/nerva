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

#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/string_utility.h"
#include "fmt/format.h"
#include <limits>
#include <map>
#include <utility>

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

// Parses the argument of a string of the shape <name>(<argument>), and converts it to a double value.
inline
double parse_numeric_argument(const std::string& text)
{
  auto startpos = text.find('(');
  auto endpos = text.find(')');
  if (startpos == std::string::npos || endpos == std::string::npos || endpos <= startpos)
  {
    throw std::runtime_error("could not parse optimizer '" + text + "'");
  }
  return parse_double(text.substr(startpos + 1, endpos - startpos - 1));
};

struct function_call
{
  std::string name;
  std::map<std::string, std::string> arguments;

  function_call(std::string name_, std::map<std::string, std::string> arguments_)
   : name(std::move(name_)), arguments(std::move(arguments_))
  {}

  [[nodiscard]] bool has_key(const std::string& key) const
  {
    return arguments.find(key) != arguments.end();
  }

  [[nodiscard]] std::string get_value(const std::string& key) const
  {
    auto i = arguments.find(key);
    if (i == arguments.end() && arguments.size() == 1)
    {
      i = arguments.find("");
    }
    if (i != arguments.end())
    {
      return i->second;
    }
    return ""; // return the empty string to indicate nothing was found
  }

  [[nodiscard]] scalar as_scalar(const std::string& key, float default_value = std::numeric_limits<scalar>::quiet_NaN()) const
  {
    std::string value = get_value(key);
    if (!value.empty())
    {
      return parse_scalar(value);
    }
    if (!std::isnan(default_value))
    {
      return default_value;
    }
    throw std::runtime_error(fmt::format("Could not find an argument named \"{}\"", key));
  }

  [[nodiscard]] std::string as_string(const std::string& key, const std::string& default_value = "") const
  {
    std::string value = get_value(key);
    if (!value.empty())
    {
      return value;
    }
    if (!default_value.empty())
    {
      return default_value;
    }
    throw std::runtime_error(fmt::format("Could not find an argument named \"{}\"", key));
  }
};

// Parse a string of the shape "NAME(key1=value1, key2=value2, ...)".
// If there are no arguments the parentheses may be omitted.
// If there is only one parameter, it is allowed to pass "NAME(value)" instead of "NAME(key=value)"
inline
function_call parse_function_call(std::string text)
{
  utilities::trim(text);

  auto error = [&text]()
  {
    throw std::runtime_error(fmt::format("Could not parse function call \"{}\"", text));
  };

  std::string name;
  std::map<std::string, std::string> arguments;

  std::smatch m;

  // no parentheses
  bool result = std::regex_match(text, m, std::regex(R"((\w+$))"));
  if (result)
  {
    name = m[1];
    return {name, arguments};
  }

  // with parentheses
  result = std::regex_match(text, m, std::regex(R"((\w+)\((.*?)\))"));
  if (result)
  {
    name = m[1];
    std::vector<std::string> args = regex_split(m[2], ",");

    if (args.size() == 1 && args.front().find('=') == std::string::npos)
    {
      // NAME(value)
      auto value = utilities::trim_copy(args.front());
      arguments[""] = value;
      return {name, arguments};
    }
    else
    {
      // NAME(key1=value1, ...)
      for (const auto& arg: args)
      {
        auto words = regex_split(arg, R"(\s*=\s*)");
        if (words.size() != 2)
        {
          error();
        }
        auto key = words[0];
        auto value = words[1];
        if (const auto &[it, inserted] = arguments.emplace(key, value); !inserted)
        {
          std::cout << "Key \"" << key << "\" appears multiple times.\n";
          error();
        }
      }
      return {name, arguments};
    }
  }

  error();
  return {name, arguments};  // to silence warnings
}

} // namespace nerva::utilities

#endif // NERVA_UTILITIES_PARSE_H
