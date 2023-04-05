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

  std::size_t weight_count = support_size(W);
  NERVA_LOG(log::verbose) << fmt::format("regrowing {}/{} weights", prune_count, weight_count) << std::endl;

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
  using eigen::support_size;

  // prune elements by giving them the value NaN
  std::size_t prune_count = prune_magnitude(W, count, std::numeric_limits<Scalar>::quiet_NaN());
  if (prune_count != count)
  {
    throw std::runtime_error(fmt::format("prune_magnitude failed: pruned {} instead of {} elements", prune_count, count));
  }

  std::size_t weight_count = support_size(W);
  NERVA_LOG(log::verbose) << fmt::format("regrowing {}/{} weights", count, weight_count) << std::endl;

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
  using eigen::support_size;

  // prune elements by giving them the value NaN
  auto nan = std::numeric_limits<Scalar>::quiet_NaN();
  std::size_t negative_prune_count = prune_negative_weights(W, negative_count, nan);
  std::size_t positive_prune_count = prune_positive_weights(W, positive_count, nan);
  assert(negative_prune_count == negative_count);
  assert(positive_prune_count == positive_count);

  std::size_t weight_count = support_size(W);
  NERVA_LOG(log::verbose) << fmt::format("regrowing {}/{} weights", negative_prune_count + positive_prune_count, weight_count) << std::endl;

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
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_REGROW_H
