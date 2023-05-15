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
#include "nerva/neural_networks/batch_normalization_layer.h"
#include "nerva/neural_networks/dropout_layers.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/utilities/parse_numbers.h"
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

struct layer_builder
{
  double dropout_rate = 0;  // dropout rate of the next linear layer
  std::mt19937& rng;        // needed for creating sparse matrices

  explicit layer_builder(std::mt19937& rng_)
    : rng(rng_)
  {}

  std::shared_ptr<dense_linear_layer> make_dense_linear_layer(const std::string& layer_description, std::size_t D, std::size_t K, long batch_size)
  {
    if (layer_description == "Linear")
    {
      return std::make_shared<dense_linear_layer>(D, K, batch_size);
    }
    else if (layer_description == "Sigmoid")
    {
      return std::make_shared<dense_sigmoid_layer>(D, K, batch_size);
    }
    else if (layer_description == "ReLU")
    {
      return std::make_shared<dense_relu_layer>(D, K, batch_size);
    }
    else if (layer_description == "Softmax")
    {
      return std::make_shared<dense_softmax_layer>(D, K, batch_size);
    }
    else if (layer_description == "LogSoftmax")
    {
      return std::make_shared<dense_log_softmax_layer>(D, K, batch_size);
    }
    else if (layer_description == "HyperbolicTangent")
    {
      return std::make_shared<dense_hyperbolic_tangent_layer>(D, K, batch_size);
    }
    else if (utilities::starts_with(layer_description, "AllRelu"))
    {
      scalar alpha = parse_scalar_argument(layer_description);
      return std::make_shared<dense_trelu_layer>(alpha, D, K, batch_size);
    }
    else if (utilities::starts_with(layer_description, "LeakyRelu"))
    {
      scalar alpha = parse_scalar_argument(layer_description);
      return std::make_shared<dense_leaky_relu_layer>(alpha, D, K, batch_size);
    }
    else if (utilities::starts_with(layer_description, "TReLU"))
    {
      scalar epsilon = parse_scalar_argument(layer_description);
      return std::make_shared<dense_trelu_layer>(epsilon, D, K, batch_size);
    }
    else if (utilities::starts_with(layer_description, "SReLU"))
    {
      const auto [al, tl, ar, tr] = parse_scalar_arguments<4>(layer_description);
      return std::make_shared<dense_srelu_layer>(D, K, batch_size, al, tl, ar, tr);
    }
    throw std::runtime_error("unsupported dense layer '" + layer_description + "'");
  }

  inline
  std::shared_ptr<sparse_linear_layer> make_sparse_linear_layer(const std::string& layer_description, std::size_t D, std::size_t K, long batch_size)
  {
    if (layer_description == "Linear")
    {
      return std::make_shared<sparse_linear_layer>(D, K, batch_size);
    }
    else if (layer_description == "Sigmoid")
    {
      return std::make_shared<sparse_sigmoid_layer>(D, K, batch_size);
    }
    else if (layer_description == "ReLU")
    {
      return std::make_shared<sparse_relu_layer>(D, K, batch_size);
    }
    else if (layer_description == "Softmax")
    {
      return std::make_shared<sparse_softmax_layer>(D, K, batch_size);
    }
    else if (layer_description == "LogSoftmax")
    {
      return std::make_shared<sparse_log_softmax_layer>(D, K, batch_size);
    }
    else if (layer_description == "HyperbolicTangent")
    {
      return std::make_shared<sparse_hyperbolic_tangent_layer>(D, K, batch_size);
    }
    else if (utilities::starts_with(layer_description, "AllReLU"))
    {
      scalar alpha = parse_scalar_argument(layer_description);
      return std::make_shared<sparse_trelu_layer>(D, K, batch_size, alpha);
    }
    else if (utilities::starts_with(layer_description, "LeakyReLU"))
    {
      scalar alpha = parse_scalar_argument(layer_description);
      return std::make_shared<sparse_leaky_relu_layer>(D, K, batch_size, alpha);
    }
    else if (utilities::starts_with(layer_description, "TReLU"))
    {
      scalar epsilon = parse_scalar_argument(layer_description);
      return std::make_shared<sparse_trelu_layer>(D, K, batch_size, epsilon);
    }
    else if (utilities::starts_with(layer_description, "SReLU"))
    {
      const auto [al, tl, ar, tr] = parse_scalar_arguments<4>(layer_description);
      return std::make_shared<sparse_srelu_layer>(D, K, batch_size, al, tl, ar, tr);
    }
    throw std::runtime_error("unsupported sparse layer '" + layer_description + "'");
  }

  std::shared_ptr<neural_network_layer> make_dense_linear_dropout_layer(const std::string& layer_description, std::size_t D, std::size_t K, long batch_size, scalar dropout)
  {
    if (layer_description == "Linear")
    {
      return std::make_shared<dense_linear_dropout_layer>(D, K, batch_size, dropout);
    }
    else if (layer_description == "Sigmoid")
    {
      return std::make_shared<dense_sigmoid_dropout_layer>(D, K, batch_size, dropout);
    }
    else if (layer_description == "ReLU")
    {
      return std::make_shared<dense_relu_dropout_layer>(D, K, batch_size, dropout);
    }
    else if (layer_description == "Softmax")
    {
      return std::make_shared<dense_softmax_dropout_layer>(D, K, batch_size, dropout);
    }
    else if (layer_description == "LogSoftmax")
    {
      return std::make_shared<dense_log_softmax_dropout_layer>(D, K, batch_size, dropout);
    }
    else if (layer_description == "HyperbolicTangent")
    {
      return std::make_shared<dense_hyperbolic_tangent_dropout_layer>(D, K, batch_size, dropout);
    }
    else if (utilities::starts_with(layer_description, "AllRelu"))
    {
      scalar alpha = parse_scalar_argument(layer_description);
      return std::make_shared<dense_trelu_dropout_layer>(alpha, D, K, batch_size, dropout);
    }
    else if (utilities::starts_with(layer_description, "LeakyRelu"))
    {
      scalar alpha = parse_scalar_argument(layer_description);
      return std::make_shared<dense_leaky_relu_dropout_layer>(alpha, D, K, batch_size, dropout);
    }
    else if (utilities::starts_with(layer_description, "TReLU"))
    {
      scalar epsilon = parse_scalar_argument(layer_description);
      return std::make_shared<dense_trelu_dropout_layer>(epsilon, D, K, batch_size, dropout);
    }
    else if (utilities::starts_with(layer_description, "SReLU"))
    {
      const auto [al, tl, ar, tr] = parse_scalar_arguments<4>(layer_description);
      return std::make_shared<dense_srelu_dropout_layer>(D, K, batch_size, dropout, al, tl, ar, tr);
    }
    throw std::runtime_error("unsupported dropout layer '" + layer_description + "'");
  }

  std::shared_ptr<neural_network_layer> make_linear_layer(const std::string& layer_description,
                                                          std::size_t D,
                                                          std::size_t K,
                                                          long batch_size,
                                                          double density,
                                                          weight_initialization w,
                                                          const std::string& optimizer
                                                         )
  {
    if (dropout_rate == 0)
    {
      if (density == 1)
      {
        std::shared_ptr<dense_linear_layer> layer = make_dense_linear_layer(layer_description, D, K, batch_size);
        set_weights_and_bias(*layer, w, rng);
        set_optimizer(*layer, optimizer);
        return layer;
      }
      else
      {
        std::shared_ptr<sparse_linear_layer> layer =  make_sparse_linear_layer(layer_description, D, K, batch_size);
        set_support_random(*layer, density, rng);
        set_weights_and_bias(*layer, w, rng);
        set_optimizer(*layer, optimizer);
        return layer;
      }
    }
    else
    {
      std::shared_ptr<neural_network_layer> layer = make_dense_linear_dropout_layer(layer_description, D, K, batch_size, dropout_rate);
      std::shared_ptr<dense_linear_layer> dlayer(dynamic_cast<dense_linear_layer*>(layer.get()));
      set_weights_and_bias(*dlayer, w, rng);
      set_optimizer(*dlayer, optimizer);
      return layer;
    }
  }

  std::vector<std::shared_ptr<neural_network_layer>> build(const std::vector<std::string>& layer_specifications,
                                                           const std::vector<std::size_t>& linear_layer_sizes,
                                                           const std::vector<double>& linear_layer_densities,
                                                           const std::vector<weight_initialization>& linear_layer_weights,
                                                           const std::vector<std::string>& linear_layer_optimizers,
                                                           long batch_size)
  {
    std::vector<std::shared_ptr<neural_network_layer>> result;

    unsigned int i = 0;  // index of the linear layers
    for (const std::string& layer: layer_specifications)
    {
      if (utilities::starts_with("Dropout", layer))
      {
        auto arguments = parse_arguments(layer, 1);
        dropout_rate = parse_double(arguments.front());
      }
      else if (layer == "BatchNorm")
      {
        result.push_back(std::make_shared<dense_batch_normalization_layer>(linear_layer_sizes[i], batch_size));
      }
      else if (is_linear_layer(layer))
      {
        result.push_back(make_linear_layer(layer, linear_layer_sizes[i], linear_layer_sizes[i + 1], batch_size, linear_layer_densities[i], linear_layer_weights[i], linear_layer_optimizers[i]));
        i++;
      }
    }

    return result;
  }
};

std::vector<std::shared_ptr<neural_network_layer>> construct_layers(const std::vector<std::string>& layer_specifications,
                                                                    const std::vector<std::size_t>& linear_layer_sizes,
                                                                    const std::vector<double>& linear_layer_densities,
                                                                    const std::vector<weight_initialization>& linear_layer_weights,
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
