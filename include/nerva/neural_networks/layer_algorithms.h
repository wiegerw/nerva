// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/layer_algorithms.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

namespace nerva {

inline
void compare_sizes(const eigen::matrix& W1, const eigen::matrix& W2)
{
  if (W1.rows() != W2.rows() || W1.cols() != W2.cols())
  {
    eigen::print_numpy_matrix("W", W1);
    eigen::print_numpy_matrix("W", W2);
    throw std::runtime_error("matrix sizes do not match");
  }
}

inline
void compare_sizes(const mkl::sparse_matrix_csr<scalar>& W1, const eigen::matrix& W2)
{
  if (W1.rows() != W2.rows() || W1.cols() != W2.cols())
  {
    eigen::print_numpy_matrix("W", mkl::to_eigen(W1));
    eigen::print_numpy_matrix("W", W2);
    throw std::runtime_error("matrix sizes do not match");
  }
}

inline
auto compute_sparse_layer_densities(double overall_density,
                                    const std::vector<std::size_t>& layer_sizes,
                                    double erk_power_scale = 1) -> std::vector<double>
{
  std::vector<std::pair<std::size_t, std::size_t>> layer_shapes;
  for (std::size_t i = 0; i < layer_sizes.size() - 1; i++)
  {
    layer_shapes.emplace_back(layer_sizes[i], layer_sizes[i+1]);
  }

  std::size_t const n = layer_shapes.size(); // the number of layers

  if (overall_density == 1)
  {
    return std::vector<double>(n, 1);
  }

  std::set<std::size_t> dense_layers;
  std::vector<double> raw_probabilities(n, double(0));
  double epsilon;

  while (true)
  {
    double divisor = 0;
    double rhs = 0;
    std::fill(raw_probabilities.begin(), raw_probabilities.end(), double(0));
    for (std::size_t i = 0; i < n; i++)
    {
      auto [rows, columns] = layer_shapes[i];
      auto N = rows * columns;
      auto num_ones = N * overall_density;
      auto num_zeros = N - num_ones;
      if (dense_layers.count(i))
      {
        rhs -= num_zeros;
      }
      else
      {
        rhs += num_ones;
        raw_probabilities[i] = ((rows + columns) / (double)(rows * columns)) * std::pow(erk_power_scale, double(1));
        divisor += raw_probabilities[i] * N;
      }
    }
    epsilon = rhs / divisor;
    double const max_prob = *std::max_element(raw_probabilities.begin(), raw_probabilities.end());
    double const max_prob_one = max_prob * epsilon;
    if (max_prob_one > 1)
    {
      for (std::size_t j = 0; j < n; j++)
      {
        if (raw_probabilities[j] == max_prob)
        {
          dense_layers.insert(j);
        }
      }
    }
    else
    {
      break;
    }
  }

  // Compute the densities
  std::vector<double> densities(n, 0);
  for (std::size_t i = 0; i < n; i++)
  {
    if (dense_layers.count(i))
    {
      densities[i] = 1;
    }
    else
    {
      double const probability_one = epsilon * raw_probabilities[i];
      densities[i] = probability_one;
    }
  }
  return densities;
}

} // namespace nerva

