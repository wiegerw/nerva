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
#include "nerva/neural_networks/prune.h"
#include "nerva/neural_networks/weights.h"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace nerva {

// Remove the smallest count nonzero elements (in absolute size) and randomly add new elements for them.
// N.B. This function must accept both Eigen::Matrix and Eigen::Ref<Eigen::Matrix>>.
template <typename EigenMatrix, typename Scalar = scalar>
void regrow_threshold(EigenMatrix& W, const std::shared_ptr<weight_initializer>& init, long count, std::mt19937& rng)
{
  // prune elements by giving them the value max_scalar
  auto max_scalar = std::numeric_limits<Scalar>::max();
  long prune_count = prune_weights(W, count, max_scalar);

  assert(prune_count == count);

  // grow elements that are equal to zero
  grow(W, init, prune_count, rng, accept_zero());

  // replace max_scalar by 0
  W = W.unaryExpr([max_scalar](scalar x) { return x == max_scalar ? 0 : x; });
}

/// Prunes and regrows a given fraction of the smallest elements (in absolute value) of the matrix \a W.
/// \param W A sparse matrix
/// \param zeta The fraction of entries in \a W that will get a new value
/// \param rng A random number generator
template <typename Scalar = scalar>
void regrow_threshold(mkl::sparse_matrix_csr<Scalar>& W, const std::shared_ptr<weight_initializer>& init, scalar zeta, std::mt19937& rng)
{
  eigen::matrix W1 = mkl::to_eigen(W);
  long nonzero_count = (W1.array() != 0).count();
  long count = std::lround(zeta * static_cast<scalar>(nonzero_count));
  regrow_threshold(W1, init, count, rng);
  W = mkl::to_csr(W1);
}

template <typename EigenMatrix, typename Scalar = scalar>
void regrow_interval(EigenMatrix& W, const std::shared_ptr<weight_initializer>& init, long negative_count, long positive_count, std::mt19937& rng)
{
  // prune elements by giving them the value max_scalar
  auto max_scalar = std::numeric_limits<Scalar>::max();
  long negative_prune_count = prune_negative_weights(W, negative_count, max_scalar);
  long positive_prune_count = prune_positive_weights(W, positive_count, max_scalar);

  assert(negative_prune_count == negative_count);
  assert(positive_prune_count == positive_count);

  // grow elements that are equal to zero
  grow(W, init, negative_prune_count + positive_prune_count, rng, accept_zero());

  // replace max_scalar by 0
  W = W.unaryExpr([max_scalar](Scalar x) { return x == max_scalar ? 0 : x; });
}

/// Prunes and regrows a given fraction of the smallest elements of matrix \a W.
/// Positive and negative entries are pruned independently.
/// \param W A sparse matrix
/// \param zeta The fraction of positive and negative entries in \a W that will get a new value
/// \param rng A random number generator
template <typename Scalar = scalar>
void regrow_interval(mkl::sparse_matrix_csr<Scalar>& W, const std::shared_ptr<weight_initializer>& init, scalar zeta, std::mt19937& rng)
{
  auto W1 = mkl::to_eigen(W);
  long negative_count = std::lround(zeta * (W1.array() < 0).count());
  long positive_count = std::lround(zeta * (W1.array() > 0).count());
  regrow_interval(W1, init, negative_count, positive_count, rng);
  W = mkl::to_csr(W1);
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
/// \param W A sparse matrix
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
