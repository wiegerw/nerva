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
#include "nerva/neural_networks/grow.h"
#include "nerva/neural_networks/prune.h"
#include "nerva/neural_networks/weights.h"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace nerva {

/// Prunes `count` nonzero elements with the smallest magnitude and randomly add new elements for them.
/// \tparam Matrix A matrix type (eigen::matrix or mkl::sparse_matrix_csr)
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param rng A random number generator
template <typename Matrix>
void regrow_threshold(Matrix& W, const std::shared_ptr<weight_initializer>& init, std::size_t count, std::mt19937& rng)
{
  using Scalar = typename Matrix::Scalar;

  // prune elements by giving them the value NaN
  auto nan = std::numeric_limits<Scalar>::quiet_NaN();
  std::size_t prune_count = prune_weights(W, count, nan);
  assert(prune_count == count);

  // grow elements that are equal to zero
  grow(W, init, prune_count, rng);
}

/// Prunes and regrows a given fraction of the smallest elements (in absolute value) of the matrix \a W.
/// \tparam Matrix A matrix type (eigen::matrix or mkl::sparse_matrix_csr)
/// \param W A matrix
/// \param zeta The fraction of entries in \a W that will get a new value
/// \param rng A random number generator
template <typename Matrix>
void regrow_threshold(Matrix& W, const std::shared_ptr<weight_initializer>& init, scalar zeta, std::mt19937& rng)
{
  std::size_t count = std::lround(zeta * static_cast<scalar>(support_size(W)));
  regrow_threshold(W, init, count, rng);
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
  grow(W, init, negative_prune_count + positive_prune_count, rng);
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

/// Prunes and regrows a given fraction of the smallest elements of matrix \a W.
/// Positive and negative entries are pruned independently.
/// \param layer A sparse linear layer
/// \param zeta The fraction of positive and negative entries in \a W that will get a new value
/// \param rng A random number generator
template <typename Scalar = scalar>
void regrow(sparse_linear_layer& layer, weight_initialization w, scalar zeta, bool separate_positive_negative, std::mt19937& rng)
{
  auto init = create_weight_initializer(layer.W, w, rng);
  if (separate_positive_negative)
  {
    regrow_interval(layer.W, init, zeta, rng);
  }
  else
  {
    regrow_threshold(layer.W, init, zeta, rng);
  }
  layer.reset_support();
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_REGROW_H
