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
#include <memory>

namespace nerva {

inline
bool is_linear_layer(char c)
{
  return std::string("LRSYZ").find(c) != std::string::npos;
}

inline
std::shared_ptr<neural_network_layer> parse_dense_layer(char c, std::size_t D, std::size_t K, const mlp_options& options, std::mt19937& rng)
{
  if (c == 'B')
  {
    return std::make_shared<dense_batch_normalization_layer>(D, options.batch_size);
  }

  if (options.dropout == scalar(0))
  {
    if (c == 'L')
    {
      return std::make_shared<dense_linear_layer>(D, K, options.batch_size);
    }
    else if (c == 'R')
    {
      return std::make_shared<dense_relu_layer>(D, K, options.batch_size);
    }
    else if (c == 'S')
    {
      return std::make_shared<dense_sigmoid_layer>(D, K, options.batch_size);
    }
    else if (c == 'Y')
    {
      return std::make_shared<dense_log_softmax_layer>(D, K, options.batch_size);
    }
    else if (c == 'Z')
    {
      return std::make_shared<dense_softmax_layer>(D, K, options.batch_size);
    }
  }
  else
  {
    if (c == 'L')
    {
      return std::make_shared<dense_linear_dropout_layer>(D, K, options.batch_size, options.dropout);
    }
    else if (c == 'R')
    {
      return std::make_shared<dense_relu_dropout_layer>(D, K, options.batch_size, options.dropout);
    }
    else if (c == 'S')
    {
      return std::make_shared<dense_sigmoid_dropout_layer>(D, K, options.batch_size, options.dropout);
    }
  }

  throw std::runtime_error(std::string("unknown layer type '") + c + "'");
}

inline
std::shared_ptr<neural_network_layer> parse_sparse_layer(char c, std::size_t D, std::size_t K, double density, const mlp_options& options, std::mt19937& rng)
{
  if (c == 'L')
  {
    auto layer = std::make_shared<sparse_linear_layer>(D, K, options.batch_size);
    initialize_sparse_weights<scalar>(*layer, density, rng);
    return layer;
  }
  else if (c == 'R')
  {
    auto layer = std::make_shared<sparse_relu_layer>(D, K, options.batch_size);
    initialize_sparse_weights<scalar>(*layer, density, rng);
    return layer;
  }
  else if (c == 'S')
  {
    auto layer = std::make_shared<sparse_sigmoid_layer>(D, K, options.batch_size);
    initialize_sparse_weights<scalar>(*layer, density, rng);
    return layer;
  }
  else if (c == 'Y')
  {
    auto layer = std::make_shared<sparse_log_softmax_layer>(D, K, options.batch_size);
    initialize_sparse_weights<scalar>(*layer, density, rng);
    return layer;
  }
  else if (c == 'Z')
  {
    auto layer = std::make_shared<sparse_softmax_layer>(D, K, options.batch_size);
    initialize_sparse_weights<scalar>(*layer, density, rng);
    return layer;
  }

  throw std::runtime_error(std::string("unknown layer type '") + c + "'");
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_PARSE_LAYER_H
