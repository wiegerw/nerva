// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/prune.h
/// \brief Prune functions for sparse MKL matrices.

#ifndef NERVA_NEURAL_NETWORKS_PRUNE_H
#define NERVA_NEURAL_NETWORKS_PRUNE_H

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/functions.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/scalar.h"
#include "nerva/utilities/algorithms.h"

namespace nerva {

namespace detail {

/// Generic version of `std::nth_element` applied to accepted elements of a sequence [first, last).
/// \param first an iterator
/// \param last an iterator
/// \param k an index in the range `0, ..., A.size() - 1)`
/// \param accept a predicate function; only values \c x for which `accept(x) == true` are considered
/// \param comp comparison function object
/// \return A pair `(value, m)` with \c value the value of the element with index \c n, if the elements were
/// sorted according to \c comp, and \c m the number of accepted elements equal to \c value in the range `0, ..., k-1`.
template <typename FwdIt, typename Accept=accept_nonzero, typename Scalar = scalar, typename Compare = std::less<Scalar>>
std::pair<scalar, std::size_t> nth_element(FwdIt first, FwdIt last, std::size_t k, Accept accept = Accept(), Compare comp = Compare())
{
  // copy the non-zero entries of A
  std::vector<Scalar> values;
  for (auto i = first; i != last; ++i)
  {
    auto value = *i;
    if (accept(value))
    {
      values.push_back(value);
    }
  }

  auto kth_element = values.begin() + k;
  std::nth_element(values.begin(), kth_element, values.end(), comp);

  Scalar value = *kth_element;
  std::size_t num_copies = 1 + std::count_if(values.begin(), kth_element, [value, comp](Scalar x) { return !comp(x, value); });
  return {value, num_copies};
}

/// Generic version of `std::nth_element` applied to accepted elements of a sequence [first, last).
/// \param first an iterator
/// \param last an iterator
/// \param k an index in the range `0, ..., A.size() - 1)`
/// \param accept a predicate function; only values \c x for which `accept(x) == true` are considered
/// \param comp comparison function object
/// \return A pair `(value, m)` with \c value the value of the element with index \c n, if the elements were
/// sorted according to \c comp, and \c m the number of accepted elements equal to \c value in the range `0, ..., k-1`.
template <typename Accept=accept_nonzero, typename Scalar = scalar, typename Compare = std::less<Scalar>>
std::pair<scalar, std::size_t> nth_element(const mkl::sparse_matrix_csr<Scalar>& A, std::size_t k, Accept accept = Accept(), Compare comp = Compare())
{
  const auto& values = A.values();
  return detail::nth_element(values.begin(), values.end(), k, accept, comp);
}

/// Overwrites entries in the sequence [first, last) that satisfy the predicate \a accept with a given value.
/// This function is used as a building block for pruning functions.
/// \param first An iterator
/// \param last An iterator
/// \param accept A predicate that determines if an element is pruned
/// \param value The value that is assigned to pruned elements
/// \return The number of entries that were pruned
template <typename FwdIt, typename Accept, typename T = scalar>
std::size_t prune(FwdIt first, FwdIt last, Accept accept, T value = 0)
{
  std::size_t count = 0;
  for (auto i = first; i != last; ++i)
  {
    if (accept(*i))
    {
      *i = value;
      count++;
    }
  }
  return count;
}

} // namespace detail

/// Overwrites entries `A[i,j]` that satisfy the predicate \a accept with a given value.
/// This function is used as a building block for pruning functions.
/// \param A A matrix
/// \param accept A predicate that determines if an element is pruned
/// \param value The value that is assigned to pruned elements
/// \return The number of entries that were pruned
template <typename Accept, typename T = scalar>
std::size_t prune(mkl::sparse_matrix_csr<T>& A, Accept accept, T value = 0)
{
  auto& values = A.values();
  return detail::prune(values.begin(), values.end(), accept, value);
}

/// Replaces the smallest \a count elements (in absolute value) from the matrix \a A
/// \tparam Matrix eigen::matrix or mkl::sparse_matrix_csr
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Scalar>
std::size_t prune_weights(mkl::sparse_matrix_csr<Scalar>& A, std::size_t count, Scalar value = 0)
{
  if (count == 0)
  {
    return 0;
  }
  Scalar threshold;             // the threshold value corresponding with count elements
  std::size_t threshold_count;  // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_all(), compare_less_absolute());
  threshold = std::fabs(threshold);
  accept_absolute_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

/// Replaces the smallest \a count elements (in absolute value) of the matrix \a A by a given value
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |support(A)|) )
template <typename T>
std::size_t prune_positive_weights(mkl::sparse_matrix_csr<T>& A, std::size_t count, T value = 0)
{
  if (count == 0)
  {
    return 0;
  }
  T threshold;                   // the threshold value corresponding with count elements
  std::size_t threshold_count;   // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_positive(), std::less<>());
  accept_positive_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

/// Prunes the smallest \a count negative elements of the matrix \a A
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |support(A)|) )
template <typename T>
std::size_t prune_negative_weights(mkl::sparse_matrix_csr<T>& A, std::size_t count, T value = 0)
{
  if (count == 0)
  {
    return 0;
  }
  T threshold;                   // the threshold value corresponding with count elements
  std::size_t threshold_count;   // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_negative(), std::greater<>());
  accept_negative_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_PRUNE_H
