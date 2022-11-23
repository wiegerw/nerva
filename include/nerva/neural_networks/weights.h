// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/weights.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_WEIGHTS_H
#define NERVA_NEURAL_NETWORKS_WEIGHTS_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/matrix.h"
#include "nerva/neural_networks/mkl_matrix.h"
#include <random>

namespace nerva {

struct initialize_weights_naive
{
  scalar low;
  scalar high;

  explicit initialize_weights_naive(scalar low_ = 0.0, scalar high_ = 1.0)
    : low(low_), high(high_)
  {}

  template <typename Matrix, typename RandomNumberGenerator>
  void operator()(Matrix& W, eigen::vector& b, RandomNumberGenerator rng) const
  {
    std::uniform_real_distribution<scalar> dist(low, high);
    auto f = [&dist, &rng]() { return dist(rng); };
    initialize_matrix(W, f);
    b = eigen::vector::NullaryExpr(b.size(), dist(rng));
  }
};

struct initialize_weights_uniform
{
  scalar low;
  scalar high;

  explicit initialize_weights_uniform(scalar low_ = -1.0, scalar high_ = 1.0)
      : low(low_), high(high_)
  {}

  template <typename Matrix, typename RandomNumberGenerator>
  void operator()(Matrix& W, eigen::vector& b, RandomNumberGenerator rng) const
  {
    std::uniform_real_distribution<scalar> dist(low, high);
    auto f = [&dist, &rng]() { return dist(rng); };
    initialize_matrix(W, f);
    b = eigen::vector::Zero(b.size());
  }
};

struct initialize_weights_xavier
{
  template <typename Matrix, typename RandomNumberGenerator>
  void operator()(Matrix& W, eigen::vector& b, RandomNumberGenerator rng) const
  {
    auto n = W.cols(); // # inputs
    scalar x = scalar(1.0) / std::sqrt(scalar(n));
    std::uniform_real_distribution<scalar> dist(-x, x);
    auto f = [&dist, &rng]() { return dist(rng); };
    initialize_matrix(W, f);
    b = eigen::vector::Zero(b.size());
  }
};

struct initialize_weights_normalized_xavier
{
  template <typename Matrix, typename RandomNumberGenerator>
  void operator()(Matrix& W, eigen::vector& b, RandomNumberGenerator rng) const
  {
    auto m = W.rows(); // # outputs
    auto n = W.cols(); // # inputs
    scalar x = std::sqrt(scalar(6.0)) / std::sqrt(scalar(m + n));
    std::uniform_real_distribution<scalar> dist(-x, x);
    auto f = [&dist, &rng]() { return dist(rng); };
    initialize_matrix(W, f);
    b = eigen::vector::Zero(b.size());
  }
};

struct initialize_weights_he
{
  template <typename Matrix, typename RandomNumberGenerator>
  void operator()(Matrix& W, eigen::vector& b, RandomNumberGenerator rng) const
  {
    auto n = W.cols(); // # inputs
    scalar mean = 0.0;
    scalar std = std::sqrt(scalar(2.0) / scalar(n));
    std::normal_distribution<scalar> dist(mean, std);
    auto f = [&dist, &rng]() { return dist(rng); };
    initialize_matrix(W, f);
    b = eigen::vector::NullaryExpr(b.size(), [&dist, &rng]() { return dist(rng); });
  }
};

enum class weight_initialization
{
  default_,
  he,
  xavier,
  xavier_normalized,
  uniform,
  pytorch,
  tensorflow
};

inline
std::ostream& operator<<(std::ostream& out, weight_initialization x)
{
  switch (x)
  {
    case weight_initialization::default_: out << "default"; break;
    case weight_initialization::he: out << "he"; break;
    case weight_initialization::xavier: out << "xavier"; break;
    case weight_initialization::xavier_normalized: out << "xavier-normalized"; break;
    case weight_initialization::uniform: out << "uniform"; break;
    case weight_initialization::pytorch: out << "pytorch"; break;
    case weight_initialization::tensorflow: out << "tensorflow"; break;
  }
  return out;
}

inline
weight_initialization parse_weight_initialization(const std::string& text)
{
  if (text == "default")
  {
    return weight_initialization::default_;
  }
  else if (text == "he")
  {
    return weight_initialization::he;
  }
  else if (text == "xavier")
  {
    return weight_initialization::xavier;
  }
  else if (text == "xavier-normalized")
  {
    return weight_initialization::xavier_normalized;
  }
  else if (text == "uniform")
  {
    return weight_initialization::uniform;
  }
  else if (text == "pytorch")
  {
    return weight_initialization::pytorch;
  }
  else if (text == "tensorflow")
  {
    return weight_initialization::tensorflow;
  }
  throw std::runtime_error("Error: could not parse weight initialization '" + text + "'");
}

template <typename Matrix, typename RandomNumberGenerator>
void initialize_weights(weight_initialization w, Matrix& W, eigen::vector& b, RandomNumberGenerator rng)
{
  switch(w)
  {
    case weight_initialization::he:
    {
      initialize_weights_he init;
      init(W, b, rng);
      break;
    }
    case weight_initialization::xavier:
    {
      initialize_weights_xavier init;
      init(W, b, rng);
      break;
    }
    case weight_initialization::xavier_normalized:
    {
      initialize_weights_normalized_xavier init;
      init(W, b, rng);
      break;
    }
    case weight_initialization::uniform:
    case weight_initialization::default_:
    case weight_initialization::pytorch:
    case weight_initialization::tensorflow:
    {
      initialize_weights_uniform init;
      init(W, b, rng);
      break;
    }
  }
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_WEIGHTS_H
