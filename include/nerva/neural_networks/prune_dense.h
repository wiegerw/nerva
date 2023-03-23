// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/prune_dense.h
/// \brief Prune functions for dense Eigen matrices.

#ifndef NERVA_NEURAL_NETWORKS_PRUNE_DENSE_H
#define NERVA_NEURAL_NETWORKS_PRUNE_DENSE_H

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/functions.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/prune.h"
#include "nerva/neural_networks/scalar.h"
#include "nerva/utilities/algorithms.h"

namespace nerva {

namespace detail {

/// Generic version of `std::nth_element` applied to accepted elements of a matrix.
/// \param A a matrix
/// \param k an index in the range `0, ..., A.size() - 1)`
/// \param accept a predicate function; only values \c x for which `accept(x) == true` are considered
/// \param comp comparison function object
/// \return A pair `(value, m)` with \c value the value of the element with index \c n, if the elements were
/// sorted according to \c comp, and \c m the number of accepted elements equal to \c value in the range `0, ..., k-1`.
template <typename EigenMatrix, typename Accept=accept_nonzero, typename Scalar = scalar, typename Compare = std::less<Scalar>>
std::pair<scalar, std::size_t> nth_element(const EigenMatrix& A, std::size_t k, Accept accept = Accept(), Compare comp = Compare())
{
  auto A_flattened = A.reshaped();
  return detail::nth_element(A_flattened.begin(), A_flattened.end(), k, accept, comp);
}

} // namespace detail

/// Overwrites entries `A[i,j]` that satisfy the predicate \a accept with a given value.
/// This function is used as a building block for pruning functions.
/// \param A A matrix
/// \param accept A predicate that determines if an element is pruned
/// \param value The value that is assigned to pruned elements
/// \return The number of entries that were pruned
template <typename Matrix, typename Accept>
std::size_t prune(Matrix& A, Accept accept, typename Matrix::Scalar value = 0)
{
  long N = A.rows() * A.cols();
  auto data = A.data();
  return detail::prune(data, data + N, accept, value);
}

/// Replaces the smallest \a count elements (in absolute value) from the matrix \a A
/// \tparam Matrix eigen::matrix or mkl::sparse_matrix_csr
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Matrix>
std::size_t prune_weights(Matrix& A, std::size_t count, typename Matrix::Scalar value = 0)
{
  typename Matrix::Scalar threshold; // the threshold value corresponding with count elements
  std::size_t threshold_count;        // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_nonzero(), compare_less_absolute());
  threshold = std::fabs(threshold);
  accept_nonzero_absolute_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

/// Replaces the smallest \a count elements (in absolute value) of the matrix \a A by a given value
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Matrix>
std::size_t prune_positive_weights(Matrix& A, std::size_t count, typename Matrix::Scalar value = 0)
{
  if (count == 0)
  {
    return 0;
  }
  typename Matrix::Scalar threshold;
  std::size_t threshold_count;
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_strictly_positive());
  accept_strictly_positive_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

/// Prunes the smallest \a count negative elements of the matrix \a A
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Matrix, typename T>
std::size_t prune_negative_weights(Matrix& A, std::size_t count, T value = 0)
{
  if (count == 0)
  {
    return 0;
  }
  T threshold;
  std::size_t threshold_count;
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_strictly_negative(), std::greater<>());  // TODO: use structured bindings when moving to C++20
  accept_strictly_negative_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_PRUNE_DENSE_H
