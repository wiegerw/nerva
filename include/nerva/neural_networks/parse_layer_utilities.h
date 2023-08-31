// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/parse_layer_utilities.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/string_utility.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>

namespace nerva {

inline
std::vector<double> parse_comma_separated_real_numbers(const std::string& text)
{
  std::vector<double> result;
  for (const std::string& word: utilities::regex_split(text, ","))
  {
    result.push_back(static_cast<double>(parse_double(word)));
  }
  return result;
}

inline
std::vector<std::string> parse_layers(const std::string& text)
{
  return utilities::regex_split(text, ";");
}

inline
std::vector<std::string> parse_arguments(const std::string& text, unsigned int expected_size)
{
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

inline
scalar parse_scalar_argument(const std::string& text)
{
  std::vector<std::string> arguments = parse_arguments(text, 1);
  return parse_scalar(arguments.front());
}

template <int N>
std::array<scalar, N> parse_scalar_arguments(const std::string& text)
{
  std::vector<std::string> arguments = parse_arguments(text, N);
  assert(arguments.size() == N);
  std::array<scalar, N> result;
  for (int i = 0; i < N; i++)
  {
    result[i] = parse_scalar(arguments[i]);
  }
  return result;
}

template <typename Predicate, typename T>
std::vector<T> filter_sequence(const std::vector<T>& items, Predicate pred)
{
  std::vector<T> result;
  for (const auto& item: items)
  {
    if (pred(item))
    {
      result.push_back(item);
    }
  }
  return result;
}

bool is_linear_layer(const std::string& layer_description)
{
  return layer_description != "BatchNorm";
}

inline
std::vector<std::size_t> compute_linear_layer_sizes(const std::string& linear_layer_sizes_text, const std::vector<std::string>& linear_layer_specifications)
{
  std::vector<std::size_t> linear_layer_sizes = parse_comma_separated_numbers(linear_layer_sizes_text);
  if (linear_layer_specifications.size() + 1 != linear_layer_sizes.size())
  {
    throw std::runtime_error("the number of layer sizes does not match with the number of linear layers");
  }
  return linear_layer_sizes;
}

template <typename Layer>
void set_weights_and_bias(Layer& layer, const std::string& weights, std::mt19937& rng)
{
  return set_weights_and_bias(layer, parse_weight_initialization(weights), rng);
}

} // namespace nerva
