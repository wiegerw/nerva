// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/regrow.h
/// \brief Algorithms for pruning and growing sparse matrices.

#ifndef NERVA_NEURAL_NETWORKS_REGROW_H
#define NERVA_NEURAL_NETWORKS_REGROW_H

#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/functions.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/grow.h"
#include "nerva/neural_networks/grow_dense.h"
#include "nerva/neural_networks/prune.h"
#include "nerva/neural_networks/prune_dense.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/parse_numbers.h"
#include "fmt/format.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nerva {

/// Prunes `count` nonzero elements with the smallest magnitude and randomly add new elements for them.
/// \tparam Matrix A matrix type (eigen::matrix or mkl::sparse_matrix_csr)
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param rng A random number generator
template <typename Matrix>
void regrow_threshold(Matrix& W, const std::shared_ptr<weight_initializer>& init, scalar threshold, std::mt19937& rng)
{
  using Scalar = typename Matrix::Scalar;

  // prune elements by giving them the value NaN
  std::size_t prune_count = prune_threshold(W, threshold, std::numeric_limits<Scalar>::quiet_NaN());

  // grow `prune_count` elements
  grow_random(W, init, prune_count, rng);
}

/// Prunes `count` nonzero elements with the smallest magnitude and randomly add new elements for them.
/// \tparam Matrix A matrix type (eigen::matrix or mkl::sparse_matrix_csr)
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param rng A random number generator
template <typename Matrix>
void regrow_magnitude(Matrix& W, const std::shared_ptr<weight_initializer>& init, std::size_t count, std::mt19937& rng)
{
  using Scalar = typename Matrix::Scalar;

  // prune elements by giving them the value NaN
  std::size_t prune_count = prune_magnitude(W, count, std::numeric_limits<Scalar>::quiet_NaN());
  assert(prune_count == count);

  // grow `prune_count` elements
  grow_random(W, init, prune_count, rng);
}

/// Prunes and regrows a given fraction of the smallest elements (in absolute value) of the matrix \a W.
/// \tparam Matrix A matrix type (eigen::matrix or mkl::sparse_matrix_csr)
/// \param W A matrix
/// \param zeta The fraction of entries in \a W that will get a new value
/// \param rng A random number generator
template <typename Matrix>
void regrow_magnitude(Matrix& W, const std::shared_ptr<weight_initializer>& init, scalar zeta, std::mt19937& rng)
{
  std::size_t count = std::lround(zeta * static_cast<scalar>(support_size(W)));
  regrow_magnitude(W, init, count, rng);
}

/// Prunes elements with the smallest magnitude and randomly add new elements for them.
/// Elements with positive and negative values are handled separately.
/// \tparam Matrix A matrix type (eigen::matrix or mkl::sparse_matrix_csr)
/// \param A A matrix
/// \param negative_count The number of elements with negative values to be pruned
/// \param positive_count The number of elements with positive values to be pruned
/// \param rng A random number generator
template <typename Matrix, typename Scalar = scalar>
void regrow_interval(Matrix& W, const std::shared_ptr<weight_initializer>& init, std::size_t negative_count, std::size_t positive_count, std::mt19937& rng)
{
  // prune elements by giving them the value NaN
  auto nan = std::numeric_limits<Scalar>::quiet_NaN();
  std::size_t negative_prune_count = prune_negative_weights(W, negative_count, nan);
  std::size_t positive_prune_count = prune_positive_weights(W, positive_count, nan);
  assert(negative_prune_count == negative_count);
  assert(positive_prune_count == positive_count);

  // grow elements that are equal to zero
  grow_random(W, init, negative_prune_count + positive_prune_count, rng);
}

/// Prunes and regrows a given fraction of the smallest elements of matrix \a W.
/// Positive and negative entries are pruned independently.
/// \param W A sparse matrix
/// \param zeta The fraction of positive and negative entries in \a W that will get a new value
/// \param rng A random number generator
template <typename Scalar = scalar>
void regrow_interval(mkl::sparse_matrix_csr<Scalar>& W, const std::shared_ptr<weight_initializer>& init, scalar zeta, std::mt19937& rng)
{
  std::size_t negative_count = std::lround(zeta * static_cast<scalar>(count_negative_elements(W)));
  std::size_t positive_count = std::lround(zeta * static_cast<scalar>(count_positive_elements(W)));
  regrow_interval(W, init, negative_count, positive_count, rng);
}

template <typename Matrix>
std::shared_ptr<weight_initializer> create_weight_initializer(const Matrix& W, weight_initialization w, std::mt19937& rng)
{
  switch(w)
  {
    case weight_initialization::he: return std::make_shared<he_weight_initializer>(rng, W.cols());
    case weight_initialization::xavier: return std::make_shared<xavier_weight_initializer>(rng, W.cols());
    case weight_initialization::xavier_normalized: return std::make_shared<xavier_normalized_weight_initializer>(rng, W.rows(), W.cols());
    case weight_initialization::zero: return std::make_shared<zero_weight_initializer>(rng);
    case weight_initialization::ten: return std::make_shared<ten_weight_initializer>(rng);
    default: return std::make_shared<uniform_weight_initializer>(rng);
  }
}

struct regrow_function
{
  virtual void operator()(multilayer_perceptron& M) const = 0;
};

// Operates on sparse layers only
struct prune_and_grow: public regrow_function
{
  std::shared_ptr<prune_function> prune;
  std::shared_ptr<grow_function> grow;

  prune_and_grow(std::shared_ptr<prune_function> prune_, std::shared_ptr<grow_function> grow_)
   : prune(std::move(prune_)), grow(std::move(grow_))
  {}

  void operator()(multilayer_perceptron& M) const override
  {
    std::cout << "=== REGROW ===" << std::endl;
    M.info("before");
    for (auto& layer: M.layers)
    {
      if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
      {
        std::size_t weight_count = support_size(slayer->W);
        std::size_t count = (*prune)(slayer->W);
        std::cout << fmt::format("regrowing {}/{} weights\n", count, weight_count);
        (*grow)(slayer->W, count);
      }
    }
    M.info("after");
  }
};

struct regrow_threshold_function: public regrow_function
{
  scalar threshold;
  weight_initialization w;
  std::mt19937& rng;

  regrow_threshold_function(scalar threshold_, weight_initialization w_, std::mt19937& rng_)
  : threshold(threshold_), w(w_), rng(rng_)
  {}

  void operator()(multilayer_perceptron& M) const override
  {
    for (auto& layer: M.layers)
    {
      if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
      {
        auto init = create_weight_initializer(slayer->W, w, rng);
        regrow_threshold(slayer->W, init, threshold, rng);
      }
    }
  }
};

struct regrow_magnitude_function: public regrow_function
{
  scalar zeta;
  weight_initialization w;
  std::mt19937& rng;

  regrow_magnitude_function(scalar zeta_, weight_initialization w_, std::mt19937& rng_)
    : zeta(zeta_), w(w_), rng(rng_)
  {}

  void operator()(multilayer_perceptron& M) const override
  {
    for (auto& layer: M.layers)
    {
      if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
      {
        auto init = create_weight_initializer(slayer->W, w, rng);
        regrow_magnitude(slayer->W, init, zeta, rng);
      }
    }
  }
};

struct regrow_SET_function: public regrow_function
{
  scalar zeta;
  weight_initialization w;
  std::mt19937& rng;

  regrow_SET_function(scalar zeta_, weight_initialization w_, std::mt19937& rng_)
   : zeta(zeta_), w(w_), rng(rng_)
  {}

  void operator()(multilayer_perceptron& M) const override
  {
    for (auto& layer: M.layers)
    {
      if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
      {
        auto init = create_weight_initializer(slayer->W, w, rng);
        regrow_interval(slayer->W, init, zeta, rng);
      }
    }
  }
};

inline
std::shared_ptr<regrow_function> parse_regrow_function(const std::string& prune_strategy, weight_initialization w, std::mt19937& rng)
{
  if (prune_strategy.empty())
  {
    return nullptr;
  }

  std::vector<std::string> arguments;

  arguments = utilities::parse_arguments(prune_strategy, "Magnitude", 1);
  if (!arguments.empty())
  {
    scalar zeta = parse_scalar(arguments.front());
    return std::make_shared<regrow_magnitude_function>(zeta, w, rng);
  }

  arguments = utilities::parse_arguments(prune_strategy, "Threshold", 1);
  if (!arguments.empty())
  {
    scalar threshold = parse_scalar(arguments.front());
    return std::make_shared<regrow_threshold_function>(threshold, w, rng);
  }

  arguments = utilities::parse_arguments(prune_strategy, "SET", 1);
  if (!arguments.empty())
  {
    scalar zeta = parse_scalar(arguments.front());
    return std::make_shared<regrow_SET_function>(zeta, w, rng);
  }

  throw std::runtime_error(fmt::format("unknown prune strategy {}", prune_strategy));
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_REGROW_H
