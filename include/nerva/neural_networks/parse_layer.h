// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/parse_layer.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_PARSE_LAYER_H
#define NERVA_NEURAL_NETWORKS_PARSE_LAYER_H

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
  return layer_description != "BatchNorm" && !utilities::starts_with(layer_description, "Dropout");
}

inline
std::vector<double> compute_linear_layer_densities(const std::string& densities_text, double overall_density, const std::vector<std::string>& linear_layer_specifications, const std::vector<std::size_t>& linear_layer_sizes)
{
  if (linear_layer_specifications.size() + 1 != linear_layer_sizes.size())
  {
    std::cout << "linear layers: " << print_list(linear_layer_specifications) << std::endl;
    std::cout << "layer sizes: " << print_list(linear_layer_sizes) << std::endl;
    throw std::runtime_error("the number of linear layer sizes does not match with the number of linear layers");
  }

  std::vector<double> densities = parse_comma_separated_real_numbers(densities_text);
  if (densities.empty())
  {
    if (overall_density == 1)
    {
      densities = std::vector<double>(linear_layer_specifications.size(), 1);
    }
    else
    {
      densities = compute_sparse_layer_densities(overall_density, linear_layer_sizes);
    }
  }

  if (densities.size() != linear_layer_specifications.size())
  {
    throw std::runtime_error("the number of densities does not match with the number of linear layers");
  }

  return densities;
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

struct layer_builder
{
  std::mt19937& rng;        // needed for creating sparse matrices

  explicit layer_builder(std::mt19937& rng_)
    : rng(rng_)
  {}

  std::shared_ptr<dense_linear_layer> make_dense_linear_layer(const std::string& name,
                                                              const std::map<std::string, std::string>& arguments,
                                                              std::size_t D,
                                                              std::size_t K,
                                                              long N,
                                                              const std::string& weights,
                                                              const std::string& optimizer
                                                              )
  {
    if (name == "Linear")
    {
      auto layer = std::make_shared<dense_linear_layer>(D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "Sigmoid")
    {
      auto layer = std::make_shared<dense_sigmoid_layer>(D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "ReLU")
    {
      auto layer = std::make_shared<dense_relu_layer>(D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "Softmax")
    {
      auto layer = std::make_shared<dense_softmax_layer>(D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "LogSoftmax")
    {
      auto layer = std::make_shared<dense_log_softmax_layer>(D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "HyperbolicTangent")
    {
      auto layer = std::make_shared<dense_hyperbolic_tangent_layer>(D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "AllRelu")
    {
      scalar alpha = utilities::get_scalar_argument(arguments, "alpha");
      auto layer = std::make_shared<dense_trelu_layer>(alpha, D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "LeakyRelu")
    {
      scalar alpha = utilities::get_scalar_argument(arguments, "alpha");
      auto layer = std::make_shared<dense_leaky_relu_layer>(alpha, D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "TReLU")
    {
      scalar epsilon = utilities::get_scalar_argument(arguments, "epsilon");
      auto layer = std::make_shared<dense_trelu_layer>(epsilon, D, K, N);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "SReLU")
    {
      scalar al = utilities::get_scalar_argument(arguments, "al", 0);
      scalar tl = utilities::get_scalar_argument(arguments, "tl", 0);
      scalar ar = utilities::get_scalar_argument(arguments, "ar", 0);
      scalar tr = utilities::get_scalar_argument(arguments, "tr", 1);
      auto layer = std::make_shared<dense_srelu_layer>(D, K, N, al, tl, ar, tr);
      set_weights_and_bias(*layer, weights, rng);
      set_srelu_layer_optimizer(*layer, optimizer);
      return layer;
    }
    throw std::runtime_error("unsupported dense layer '" + name + "'");
  }

  inline
  std::shared_ptr<sparse_linear_layer> make_sparse_linear_layer(const std::string& name,
                                                                const std::map<std::string, std::string>& arguments,
                                                                std::size_t D,
                                                                std::size_t K,
                                                                long N,
                                                                scalar density,
                                                                const std::string& weights,
                                                                const std::string& optimizer)
  {
    if (name == "Linear")
    {
      auto layer = std::make_shared<sparse_linear_layer>(D, K, N);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "Sigmoid")
    {
      auto layer = std::make_shared<sparse_sigmoid_layer>(D, K, N);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "ReLU")
    {
      auto layer = std::make_shared<sparse_relu_layer>(D, K, N);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "Softmax")
    {
      auto layer = std::make_shared<sparse_softmax_layer>(D, K, N);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "LogSoftmax")
    {
      auto layer = std::make_shared<sparse_log_softmax_layer>(D, K, N);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "HyperbolicTangent")
    {
      auto layer = std::make_shared<sparse_hyperbolic_tangent_layer>(D, K, N);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "AllReLU")
    {
      scalar alpha = utilities::get_scalar_argument(arguments, "alpha");
      auto layer = std::make_shared<sparse_trelu_layer>(D, K, N, alpha);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "LeakyReLU")
    {
      scalar alpha = utilities::get_scalar_argument(arguments, "alpha");
      auto layer = std::make_shared<sparse_leaky_relu_layer>(D, K, N, alpha);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "TReLU")
    {
      scalar epsilon = utilities::get_scalar_argument(arguments, "epsilon");
      auto layer = std::make_shared<sparse_trelu_layer>(D, K, N, epsilon);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "SReLU")
    {
      scalar al = utilities::get_scalar_argument(arguments, "al", 0);
      scalar tl = utilities::get_scalar_argument(arguments, "tl", 0);
      scalar ar = utilities::get_scalar_argument(arguments, "ar", 0);
      scalar tr = utilities::get_scalar_argument(arguments, "tr", 1);
      auto layer = std::make_shared<sparse_srelu_layer>(D, K, N, al, tl, ar, tr);
      set_support_random(*layer, density, rng);
      set_weights_and_bias(*layer, weights, rng);
      set_srelu_layer_optimizer(*layer, optimizer);
      return layer;
    }
    throw std::runtime_error("unsupported sparse layer '" + name + "'");
  }

  std::shared_ptr<neural_network_layer> make_dense_linear_dropout_layer(const std::string& name,
                                                                        const std::map<std::string, std::string>& arguments,
                                                                        std::size_t D,
                                                                        std::size_t K,
                                                                        long N,
                                                                        scalar dropout,
                                                                        const std::string& weights,
                                                                        const std::string& optimizer)
  {
    if (name == "Linear")
    {
      auto layer = std::make_shared<dense_linear_dropout_layer>(D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "Sigmoid")
    {
      auto layer = std::make_shared<dense_sigmoid_dropout_layer>(D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "ReLU")
    {
      auto layer = std::make_shared<dense_relu_dropout_layer>(D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "Softmax")
    {
      auto layer = std::make_shared<dense_softmax_dropout_layer>(D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "LogSoftmax")
    {
      auto layer = std::make_shared<dense_log_softmax_dropout_layer>(D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "HyperbolicTangent")
    {
      auto layer = std::make_shared<dense_hyperbolic_tangent_dropout_layer>(D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "AllRelu")
    {
      scalar alpha = utilities::get_scalar_argument(arguments, "alpha");
      auto layer = std::make_shared<dense_trelu_dropout_layer>(alpha, D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "LeakyRelu")
    {
      scalar alpha = utilities::get_scalar_argument(arguments, "alpha");
      auto layer = std::make_shared<dense_leaky_relu_dropout_layer>(alpha, D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "TReLU")
    {
      scalar epsilon = utilities::get_scalar_argument(arguments, "epsilon");
      auto layer = std::make_shared<dense_trelu_dropout_layer>(epsilon, D, K, N, dropout);
      set_weights_and_bias(*layer, weights, rng);
      set_linear_layer_optimizer(*layer, optimizer);
      return layer;
    }
    else if (name == "SReLU")
    {
      scalar al = utilities::get_scalar_argument(arguments, "al", 0);
      scalar tl = utilities::get_scalar_argument(arguments, "tl", 0);
      scalar ar = utilities::get_scalar_argument(arguments, "ar", 0);
      scalar tr = utilities::get_scalar_argument(arguments, "tr", 1);
      auto layer = std::make_shared<dense_srelu_dropout_layer>(D, K, N, dropout, al, tl, ar, tr);
      set_weights_and_bias(*layer, weights, rng);
      set_srelu_layer_optimizer(*layer, optimizer);
      return layer;
    }
    throw std::runtime_error("unsupported dropout layer '" + name + "'");
  }

  std::shared_ptr<neural_network_layer> make_linear_layer(const std::string& name,
                                                          const std::map<std::string, std::string>& arguments,
                                                          std::size_t input_size,
                                                          std::size_t output_size,
                                                          long batch_size,
                                                          scalar density,
                                                          const std::string& weights,
                                                          const std::string& optimizer
                                                         )
  {
    auto D = input_size;
    auto K = output_size;
    auto N = batch_size;

    auto it = arguments.find("dropout");
    scalar dropout_rate = it == arguments.end() ? scalar(0) : parse_scalar(it->second);

    if (dropout_rate == 0)
    {
      if (density == 1)
      {
        return make_dense_linear_layer(name, arguments, D, K, N, weights, optimizer);
      }
      else
      {
        return make_sparse_linear_layer(name, arguments, D, K, N, density, weights, optimizer);
      }
    }
    else
    {
      return make_dense_linear_dropout_layer(name, arguments, D, K, N, dropout_rate, weights, optimizer);
    }
  }

  std::vector<std::shared_ptr<neural_network_layer>> build(const std::vector<std::string>& layer_specifications,
                                                           const std::vector<std::size_t>& linear_layer_sizes,
                                                           const std::vector<double>& linear_layer_densities,
                                                           const std::vector<std::string>& linear_layer_weights,
                                                           const std::vector<std::string>& optimizers,
                                                           long batch_size)
  {
    std::vector<std::shared_ptr<neural_network_layer>> result;

    unsigned int layer_index = 0;
    unsigned int optimizer_index = 0;
    for (const std::string& layer: layer_specifications)
    {
      auto [name, arguments] = utilities::parse_function_call(layer);

      if (name == "BatchNorm")
      {
        auto blayer = std::make_shared<dense_batch_normalization_layer>(linear_layer_sizes[layer_index++], batch_size);
        set_batch_normalization_layer_optimizer(*blayer, optimizers[optimizer_index++]);
        result.push_back(blayer);
      }
      else
      {
        auto D = linear_layer_sizes[layer_index];
        auto K = linear_layer_sizes[layer_index + 1];
        auto N = batch_size;
        result.push_back(make_linear_layer(name, arguments, D, K, N, static_cast<scalar>(linear_layer_densities[layer_index]), linear_layer_weights[layer_index], optimizers[optimizer_index++]));
        layer_index++;
      }
    }

    return result;
  }
};

std::vector<std::shared_ptr<neural_network_layer>> construct_layers(const std::vector<std::string>& layer_specifications,
                                                                    const std::vector<std::size_t>& linear_layer_sizes,
                                                                    const std::vector<double>& linear_layer_densities,
                                                                    const std::vector<std::string>& linear_layer_weights,
                                                                    const std::vector<std::string>& linear_layer_optimizers,
                                                                    long batch_size,
                                                                    std::mt19937& rng
)
{
  layer_builder builder(rng);
  return builder.build(layer_specifications, linear_layer_sizes, linear_layer_densities, linear_layer_weights, linear_layer_optimizers, batch_size);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_PARSE_LAYER_H
