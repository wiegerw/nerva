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
#include "nerva/utilities/parse.h"
#include "nerva/utilities/parse_numbers.h"

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
/// \param first
/// \param last
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param accept Only weights that satisfy `accept` are pruned
/// \param value The value that is assigned to pruned elements (default 0)
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Scalar, typename Accept, typename FwdIt>
std::size_t prune_magnitude_with_threshold(FwdIt first, FwdIt last, std::size_t count, Accept accept = Accept(), Scalar value = 0)
{
  if (count == 0)
  {
    return 0;
  }

  Scalar threshold;             // the threshold value corresponding with count elements
  std::size_t threshold_count;  // the number of copies of threshold that should be accepted
  std::tie(threshold, threshold_count) = detail::nth_element(first, last, count - 1, accept, less_magnitude());
  threshold = std::fabs(threshold);

  auto accept_prune = [threshold, &threshold_count, accept](auto x)
  {
    auto x_ = std::fabs(x);
    return accept(x) && (x_ < threshold || (x_ == threshold && detail::decrement_count(threshold_count)));
  };

  return detail::prune(first, last, accept_prune, value);
}

} // namespace detail

/// Replaces the smallest \a count elements (in absolute value) from the matrix \a A
/// \tparam Matrix eigen::matrix or mkl::sparse_matrix_csr
/// \param A A matrix
/// \param count The number of elements to be pruned
/// \param accept Only weights that satisfy `accept` are pruned
/// \param value The value that is assigned to pruned elements (default 0)
/// \return The actual number of elements that have been pruned (i.e. min(count, |A|) )
template <typename Scalar, typename Accept>
std::size_t prune_magnitude_with_threshold(mkl::sparse_matrix_csr<Scalar>& A, std::size_t count, Accept accept = Accept(), Scalar value = 0)
{
  auto& values = A.values();
  return detail::prune_magnitude_with_threshold(values.begin(), values.end(), count, accept, value);
}

/// Replaces the smallest \a count elements (in absolute value) from the matrix \a A
/// \param A A matrix
/// \param count The maximum number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The number of elements that have been pruned
template <typename Scalar>
std::size_t prune_magnitude(mkl::sparse_matrix_csr<Scalar>& A, std::size_t count, Scalar value = 0)
{
  return prune_magnitude_with_threshold(A, count, accept_all(), value);
}

/// Replaces the smallest \a count positive elements of the matrix \a A by a given value
/// \param A A matrix
/// \param count The maximum number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The number of elements that have been pruned
template <typename T>
std::size_t prune_positive_weights(mkl::sparse_matrix_csr<T>& A, std::size_t count, T value = 0)
{
  return prune_magnitude_with_threshold(A, count, accept_positive(), value);
}

/// Replaces the smallest \a count negative elements of the matrix \a A by a given value
/// \param A A matrix
/// \param count The maximum number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The number of elements that have been pruned
template <typename T>
std::size_t prune_negative_weights(mkl::sparse_matrix_csr<T>& A, std::size_t count, T value = 0)
{
  return prune_magnitude_with_threshold(A, count, accept_negative(), value);
}

/// Replaces all elements \a x with `|x| <= threshold` from the matrix \a A
/// \param A A matrix
/// \param count The maximum number of elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The number of elements that have been pruned
template <typename Scalar>
std::size_t prune_threshold(mkl::sparse_matrix_csr<Scalar>& A, scalar threshold, Scalar value = 0)
{
  return prune_magnitude_with_threshold(A, std::numeric_limits<std::size_t>::max(), accept_threshold(threshold), value);
}

/// Replaces a fraction of positive and negative elements from the matrix \a A
/// \param A A matrix
/// \param zeta The fraction of positive and negative elements to be pruned
/// \param value The value that is assigned to the pruned elements (default 0)
/// \return The number of elements that have been pruned
template <typename Scalar>
std::size_t prune_SET(mkl::sparse_matrix_csr<Scalar>& A, scalar zeta, Scalar value = 0)
{
  std::size_t negative_count = std::lround(zeta * mkl::count_negative_elements(A));
  std::size_t positive_count = std::lround(zeta * mkl::count_positive_elements(A));
  std::size_t count = prune_positive_weights(A, positive_count, value);
  count += prune_negative_weights(A, negative_count, value);
  return count;
}

struct prune_function
{
  virtual std::size_t operator()(mkl::sparse_matrix_csr<scalar>& W) const = 0;
};

struct prune_magnitude_function: public prune_function
{
  scalar zeta;

  explicit prune_magnitude_function(scalar zeta_)
    : zeta(zeta_)
  {}

  std::size_t operator()(mkl::sparse_matrix_csr<scalar>& W) const override
  {
    std::size_t count = std::lround(zeta * mkl::support_size(W));
    return prune_magnitude(W, count, std::numeric_limits<scalar>::quiet_NaN());
  }
};

struct prune_threshold_function: public prune_function
{
  scalar threshold;

  explicit prune_threshold_function(scalar threshold_)
    : threshold(threshold_)
  {}

  std::size_t operator()(mkl::sparse_matrix_csr<scalar>& W) const override
  {
    return prune_threshold(W, threshold, std::numeric_limits<scalar>::quiet_NaN());
  }
};

struct prune_SET_function: public prune_function
{
  scalar zeta;

  explicit prune_SET_function(scalar zeta_)
    : zeta(zeta_)
  {}

  std::size_t operator()(mkl::sparse_matrix_csr<scalar>& W) const override
  {
    return prune_SET(W, zeta, std::numeric_limits<scalar>::quiet_NaN());
  }
};

inline
std::shared_ptr<prune_function> parse_prune_function(const std::string& strategy)
{
  if (utilities::trim_copy(strategy).empty())
  {
    return nullptr;
  }

  std::vector<std::string> arguments;

  arguments = utilities::parse_arguments(strategy, "Magnitude", 1);
  if (!arguments.empty())
  {
    scalar zeta = parse_scalar(arguments.front());
    return std::make_shared<prune_magnitude_function>(zeta);
  }

  arguments = utilities::parse_arguments(strategy, "Threshold", 1);
  if (!arguments.empty())
  {
    scalar threshold = parse_scalar(arguments.front());
    return std::make_shared<prune_threshold_function>(threshold);
  }

  arguments = utilities::parse_arguments(strategy, "SET", 1);
  if (!arguments.empty())
  {
    scalar zeta = parse_scalar(arguments.front());
    return std::make_shared<prune_SET_function>(zeta);
  }

  throw std::runtime_error(fmt::format("unknown prune strategy {}", strategy));
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_PRUNE_H
