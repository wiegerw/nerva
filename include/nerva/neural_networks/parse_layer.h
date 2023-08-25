// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/parse_layer.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/batch_normalization_layer_colwise.h"
#include "nerva/neural_networks/dropout_layers.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/string_utility.h"
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

namespace nerva {

// supported layers:
//
// Linear
// Sigmoid
// ReLU
// Softmax
// LogSoftmax
// HyperbolicTangent
// AllRelu(<alpha>)
// LeakyRelu(<alpha>)
// TReLU(<epsilon>)
// SReLU(<al>,<tl>,<ar>,<tr>)
//
// BatchNorm
// Dropout(<rate>)

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

inline
std::shared_ptr<dense_linear_layer> make_dense_linear_layer(std::size_t D,
                                                            std::size_t K,
                                                            long N,
                                                            const std::string& activation,
                                                            const std::string& weights,
                                                            const std::string& optimizer,
                                                            std::mt19937& rng
                                                            )
{
  auto func = utilities::parse_function_call(activation);
  if (func.name == "Linear")
  {
    auto layer = std::make_shared<dense_linear_layer>(D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "Sigmoid")
  {
    auto layer = std::make_shared<dense_sigmoid_layer>(D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "ReLU")
  {
    auto layer = std::make_shared<dense_relu_layer>(D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "Softmax")
  {
    auto layer = std::make_shared<dense_softmax_layer>(D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "LogSoftmax")
  {
    auto layer = std::make_shared<dense_log_softmax_layer>(D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "HyperbolicTangent")
  {
    auto layer = std::make_shared<dense_hyperbolic_tangent_layer>(D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "AllRelu")
  {
    scalar alpha = func.as_scalar("alpha");
    auto layer = std::make_shared<dense_trelu_layer>(alpha, D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "LeakyRelu")
  {
    scalar alpha = func.as_scalar("alpha");
    auto layer = std::make_shared<dense_leaky_relu_layer>(alpha, D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "TReLU")
  {
    scalar epsilon = func.as_scalar("epsilon");
    auto layer = std::make_shared<dense_trelu_layer>(epsilon, D, K, N);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "SReLU")
  {
    scalar al = func.as_scalar("al", 0);
    scalar tl = func.as_scalar("tl", 0);
    scalar ar = func.as_scalar("ar", 0);
    scalar tr = func.as_scalar("tr", 1);
    auto layer = std::make_shared<dense_srelu_layer>(D, K, N, al, tl, ar, tr);
    set_weights_and_bias(*layer, weights, rng);
    set_srelu_layer_optimizer(*layer, optimizer);
    return layer;
  }
  throw std::runtime_error("unsupported dense layer '" + func.name + "'");
}

inline
std::shared_ptr<sparse_linear_layer> make_sparse_linear_layer(std::size_t D,
                                                              std::size_t K,
                                                              long N,
                                                              scalar density,
                                                              const std::string& activation,
                                                              const std::string& weights,
                                                              const std::string& optimizer,
                                                              std::mt19937& rng
                                                              )
{
  auto func = utilities::parse_function_call(activation);
  if (func.name == "Linear")
  {
    auto layer = std::make_shared<sparse_linear_layer>(D, K, N);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "Sigmoid")
  {
    auto layer = std::make_shared<sparse_sigmoid_layer>(D, K, N);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "ReLU")
  {
    auto layer = std::make_shared<sparse_relu_layer>(D, K, N);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "Softmax")
  {
    auto layer = std::make_shared<sparse_softmax_layer>(D, K, N);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "LogSoftmax")
  {
    auto layer = std::make_shared<sparse_log_softmax_layer>(D, K, N);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "HyperbolicTangent")
  {
    auto layer = std::make_shared<sparse_hyperbolic_tangent_layer>(D, K, N);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "AllReLU")
  {
    scalar alpha = func.as_scalar("alpha");
    auto layer = std::make_shared<sparse_trelu_layer>(D, K, N, alpha);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "LeakyReLU")
  {
    scalar alpha = func.as_scalar("alpha");
    auto layer = std::make_shared<sparse_leaky_relu_layer>(D, K, N, alpha);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "TReLU")
  {
    scalar epsilon = func.as_scalar("epsilon");
    auto layer = std::make_shared<sparse_trelu_layer>(D, K, N, epsilon);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "SReLU")
  {
    scalar al = func.as_scalar("al", 0);
    scalar tl = func.as_scalar("tl", 0);
    scalar ar = func.as_scalar("ar", 0);
    scalar tr = func.as_scalar("tr", 1);
    auto layer = std::make_shared<sparse_srelu_layer>(D, K, N, al, tl, ar, tr);
    set_support_random(*layer, density, rng);
    set_weights_and_bias(*layer, weights, rng);
    set_srelu_layer_optimizer(*layer, optimizer);
    return layer;
  }
  throw std::runtime_error("unsupported sparse layer '" + func.name + "'");
}

inline
std::shared_ptr<neural_network_layer> make_dense_linear_dropout_layer(std::size_t D,
                                                                      std::size_t K,
                                                                      long N,
                                                                      scalar dropout,
                                                                      const std::string& activation,
                                                                      const std::string& weights,
                                                                      const std::string& optimizer,
                                                                      std::mt19937& rng
                                                                      )
{
  auto func = utilities::parse_function_call(activation);
  if (func.name == "Linear")
  {
    auto layer = std::make_shared<dense_linear_dropout_layer>(D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "Sigmoid")
  {
    auto layer = std::make_shared<dense_sigmoid_dropout_layer>(D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "ReLU")
  {
    auto layer = std::make_shared<dense_relu_dropout_layer>(D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "Softmax")
  {
    auto layer = std::make_shared<dense_softmax_dropout_layer>(D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "LogSoftmax")
  {
    auto layer = std::make_shared<dense_log_softmax_dropout_layer>(D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "HyperbolicTangent")
  {
    auto layer = std::make_shared<dense_hyperbolic_tangent_dropout_layer>(D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "AllRelu")
  {
    scalar alpha = func.as_scalar("alpha");
    auto layer = std::make_shared<dense_trelu_dropout_layer>(alpha, D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "LeakyRelu")
  {
    scalar alpha = func.as_scalar("alpha");
    auto layer = std::make_shared<dense_leaky_relu_dropout_layer>(alpha, D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "TReLU")
  {
    scalar epsilon = func.as_scalar("epsilon");
    auto layer = std::make_shared<dense_trelu_dropout_layer>(epsilon, D, K, N, dropout);
    set_weights_and_bias(*layer, weights, rng);
    set_linear_layer_optimizer(*layer, optimizer);
    return layer;
  }
  else if (func.name == "SReLU")
  {
    scalar al = func.as_scalar("al", 0);
    scalar tl = func.as_scalar("tl", 0);
    scalar ar = func.as_scalar("ar", 0);
    scalar tr = func.as_scalar("tr", 1);
    auto layer = std::make_shared<dense_srelu_dropout_layer>(D, K, N, dropout, al, tl, ar, tr);
    set_weights_and_bias(*layer, weights, rng);
    set_srelu_layer_optimizer(*layer, optimizer);
    return layer;
  }
  throw std::runtime_error("unsupported dropout layer '" + func.name + "'");
}

std::shared_ptr<neural_network_layer> make_linear_layer(std::size_t input_size,
                                                        std::size_t output_size,
                                                        long batch_size,
                                                        scalar density,
                                                        scalar dropout_rate,
                                                        const std::string& activation,
                                                        const std::string& weights,
                                                        const std::string& optimizer,
                                                        std::mt19937& rng
                                                       )
{
  auto D = input_size;
  auto K = output_size;
  auto N = batch_size;

  if (dropout_rate == 0)
  {
    if (density == 1)
    {
      return make_dense_linear_layer(D, K, N, activation, weights, optimizer, rng);
    }
    else
    {
      return make_sparse_linear_layer(D, K, N, density, activation, weights, optimizer, rng);
    }
  }
  else
  {
    return make_dense_linear_dropout_layer(D, K, N, dropout_rate, activation, weights, optimizer, rng);
  }
}

inline
std::shared_ptr<neural_network_layer> make_batch_normalization_layer(std::size_t input_size,
                                                                     long batch_size,
                                                                     const std::string& optimizer
                                                                    )
{
  auto D = input_size;
  auto layer = std::make_shared<batch_normalization_layer>(D, batch_size);
  set_batch_normalization_layer_optimizer(*layer, optimizer);
  return layer;
}

inline
void check_sizes(const std::vector<std::string>& layer_specifications,
                 const std::vector<std::size_t>& linear_layer_sizes,
                 const std::vector<double>& linear_layer_densities,
                 const std::vector<double>& linear_layer_dropouts,
                 const std::vector<std::string>& linear_layer_weights,
                 const std::vector<std::string>& optimizers
                )
{
  if (linear_layer_densities.size() != linear_layer_weights.size())
  {
    throw std::runtime_error("Size mismatch between linear_layer_densities and linear_layer_weights");
  }

  if (linear_layer_densities.size() != linear_layer_dropouts.size())
  {
    throw std::runtime_error("Size mismatch between linear_layer_densities and linear_layer_dropouts");
  }

  if (linear_layer_densities.size() + 1 != linear_layer_sizes.size())
  {
    throw std::runtime_error("Size mismatch between linear_layer_densities and linear_layer_sizes");
  }

  if (optimizers.size() != layer_specifications.size())
  {
    throw std::runtime_error("Size mismatch between optimizers and layer_specifications");
  }
}

inline
std::vector<std::shared_ptr<neural_network_layer>> make_layers(const std::vector<std::string>& layer_specifications,
                                                               const std::vector<std::size_t>& linear_layer_sizes,
                                                               const std::vector<double>& linear_layer_densities,
                                                               const std::vector<double>& linear_layer_dropouts,
                                                               const std::vector<std::string>& linear_layer_weights,
                                                               const std::vector<std::string>& optimizers,
                                                               long batch_size,
                                                               std::mt19937& rng
                                                              )
{
  check_sizes(layer_specifications, linear_layer_sizes, linear_layer_densities, linear_layer_dropouts, linear_layer_weights, optimizers);
  std::vector<std::shared_ptr<neural_network_layer>> result;

  unsigned int linear_layer_index = 0;
  unsigned int optimizer_index = 0;
  std::size_t input_size = linear_layer_sizes[0];

  for (const std::string& layer: layer_specifications)
  {
    if (layer == "BatchNorm")
    {
      auto D = input_size;
      const std::string& optimizer = optimizers[optimizer_index++];
      auto blayer = make_batch_normalization_layer(D, batch_size, optimizer);
      result.push_back(blayer);
    }
    else // linear layer
    {
      auto D = linear_layer_sizes[linear_layer_index];
      auto K = linear_layer_sizes[linear_layer_index + 1];
      auto N = batch_size;
      auto density = static_cast<scalar>(linear_layer_densities[linear_layer_index]);
      auto dropout_rate = static_cast<scalar>(linear_layer_dropouts[linear_layer_index]);
      const std::string& activation = layer;
      const std::string& weights = linear_layer_weights[linear_layer_index++];
      const std::string& optimizer = optimizers[optimizer_index++];
      auto llayer = make_linear_layer(D, K, N, density, dropout_rate, activation, weights, optimizer, rng);
      result.push_back(llayer);
      input_size = K;
    }
  }

  return result;
}

inline
std::shared_ptr<neural_network_layer> make_layer(const std::map<std::string, std::string>& m, std::mt19937& rng)
{
  std::string type = m.at("type");
  if (type == "batch-normalization")
  {
    auto D = parse_natural_number(m.at("input-size"));
    auto N = parse_natural_number<long>(m.at("batch-size"));
    const std::string& optimizer = m.at("optimizer");
    return make_batch_normalization_layer(D, N, optimizer);
  }
  else if (type == "linear")
  {
    auto D = parse_natural_number(m.at("input-size"));
    auto K = parse_natural_number(m.at("output-size"));
    auto N = parse_natural_number<long>(m.at("batch-size"));
    auto density = parse_scalar(m.at("density"));
    auto dropout_rate = parse_scalar(m.at("dropout"));
    const std::string& activation = m.at("activation");
    const std::string& weights = m.at("weights");
    const std::string& optimizer = m.at("optimizer");
    return make_linear_layer(D, K, N, density, dropout_rate, activation, weights, optimizer, rng);
  }
  throw std::runtime_error(fmt::format("Unknown layer type '{}'", type));
}

} // namespace nerva

