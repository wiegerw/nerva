// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/prune.h
/// \brief This file contains building blocks for pruning and growing (weight) matrices.

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
  long N = A.rows() * A.cols();
  const Scalar* data = A.data();

  // copy the non-zero entries of A
  std::vector<Scalar> values;
  for (long i = 0; i < N; i++)
  {
    auto value = data[i];
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
  return detail::nth_element(values.begin(), values.end(), k, accept_all(), compare_less_absolute());
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
template <typename Matrix, typename Accept, typename T = scalar>
std::size_t prune(Matrix& A, Accept accept, T value = 0)
{
  long N = A.rows() * A.cols();
  T* data = A.data();
  return detail::prune(data, data + N, accept, value);
}

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

/// Replaces the smallest \a count elements (in absolute value) from the matrix \a A
/// \tparam Matrix eigen::matrix or mkl::sparse_matrix_csr
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Scalar>
std::size_t prune_weights(mkl::sparse_matrix_csr<Scalar>& A, std::size_t count, Scalar value = 0)
{
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
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Matrix>
std::size_t prune_positive_weights(Matrix& A, std::size_t count, typename Matrix::Scalar value = 0)
{
  assert(count > 0);
  typename Matrix::Scalar threshold;
  std::size_t threshold_count;
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_strictly_positive());
  accept_strictly_positive_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

/// Replaces the smallest \a count elements (in absolute value) of the matrix \a A by a given value
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |support(A)|) )
template <typename T>
std::size_t prune_positive_weights(mkl::sparse_matrix_csr<T>& A, std::size_t count, T value = 0)
{
  assert(count > 0);
  T threshold;                     // the threshold value corresponding with count elements
  std::size_t threshold_count;   // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_positive(), std::less<>());
  accept_positive_with_threshold accept(threshold, threshold_count);
  return detail::prune(A, accept, value);
}

/// Prunes the smallest \a count negative elements of the matrix \a A
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Matrix, typename T>
std::size_t prune_negative_weights(Matrix& A, std::size_t count, T value = 0)
{
  assert(count > 0);
  T threshold;
  std::size_t threshold_count;
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_strictly_negative(), std::greater<>());  // TODO: use structured bindings when moving to C++20
  accept_strictly_negative_with_threshold accept(threshold, threshold_count);
  return prune(A, accept, value);
}

/// Prunes the smallest \a count negative elements of the matrix \a A
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \return The actual number of elements that have been pruned (i.e. min(count, |support(A)|) )
template <typename T>
std::size_t prune_negative_weights(mkl::sparse_matrix_csr<T>& A, std::size_t count, T value = 0)
{
  assert(count > 0);
  T threshold;                     // the threshold value corresponding with count elements
  std::size_t threshold_count;   // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(A, count - 1, accept_negative(), std::greater<>());
  accept_negative_with_threshold accept(threshold, threshold_count);
  return detail::prune(A, accept, value);
}

/// Gives \a count random accepted entries of \a A a new value generated by the function \a f
/// \tparam Function A function type
/// \param A A matrix
/// \param count The number of entries in \a A that will get a new value
/// \param f A weight initializer
/// \param rng A random number generator
/// \param accept A predicate that determines if an element may get a new value
template <typename Matrix, typename Accept=accept_zero, typename Scalar = scalar>
void grow(Matrix& A, const std::shared_ptr<weight_initializer>& init, std::size_t count, std::mt19937& rng, Accept accept=Accept())
{
  long N = A.rows() * A.cols();
  Scalar* data = A.data();

  // use reservoir sampling to randomly select count pointers to entries in A
  std::vector<Scalar*> selection;
  selection.reserve(count);

  long i = 0;
  for (; i < N; i++)
  {
    if (!accept(*(data + i)))
    {
      continue;
    }
    selection.push_back(data + i);
    if (selection.size() == count)
    {
      break;
    }
  }

  if (selection.size() != static_cast<std::size_t>(count))
  {
    auto available = std::count_if(data, data + N, accept);
    throw std::runtime_error("could not find " + std::to_string(count) + " positions in function grow; #available positions = " + std::to_string(available));
  }

  i += 1; // position i has already been handled

  for (; i < N; i++)
  {
    if (!accept(*(data + i)))
    {
      continue;
    }
    std::uniform_int_distribution<std::size_t> dist(0, i - 1);
    std::size_t j = dist(rng);
    if (j < count)
    {
      selection[j] = data + i;
    }
  }

  // assign new values to the chosen entries
  for (Scalar* x: selection)
  {
    *x = (*init)();
  }
}

/// Gives \a count random accepted entries of \a A a new value generated by the function \a f
/// \tparam Scalar A number type (float of double)
/// \param A A CSR matrix
/// \param init A weight initializer. The values of added elements will be initialized using \a init.
/// \param count The number of entries that will be added
/// \param rng A random number generator
/// \param remove_nan_values If true, elements with NaN values in A are removed
template <typename Scalar = scalar>
void grow(mkl::sparse_matrix_csr<Scalar>& A, const std::shared_ptr<weight_initializer>& init, std::size_t count, std::mt19937& rng, bool remove_nan_values = true)
{
  std::size_t N = A.rows() * A.cols();
  if (A.values().size() + count > N)
  {
    throw std::runtime_error("cannot grow the matrix with " + std::to_string(count) + " elements");
  }

  // components of the result matrix
  std::vector<MKL_INT> row_index;
  std::vector<MKL_INT> col_index;
  std::vector<Scalar> values;
  row_index.reserve(A.row_index().size());
  col_index.reserve(A.col_index().size());
  values.reserve(A.values().size());

  // Select k random locations outside the support of A
  std::vector<std::size_t> new_locations = reservoir_sample(count, N - A.values().size(), rng);
  auto ni = new_locations.begin();

  mkl::csr_matrix_builder<Scalar> builder(A.rows(), A.cols(), A.values().size());

  // fills the result matrix until location (i, j)
  auto fill_until_index = [&](std::size_t i, std::size_t j)
  {
    std::size_t k = i * A.cols() + j;
    while (ni != new_locations.end() && *ni < k)
    {
      std::size_t k1 = *ni;
      std::size_t i1 = k1 / A.cols();
      std::size_t j1 = k1 % A.cols();
      auto value1 = (*init)();
      builder.add_element(i1, j1, value1);
    }
  };

  mkl::traverse_elements(A, [&](long i, long j, Scalar value)
  {
    fill_until_index(i, j);
    if (!remove_nan_values || !std::isnan(value))
    {
      builder.add_element(i, j, value);
    }
  });

  A = builder.result();  // TODO: avoid unnecessary copies
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_PRUNE_H
