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
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include <random>

namespace nerva {

struct weight_initializer
{
  std::mt19937& rng;

  explicit weight_initializer(std::mt19937& rng_)
   : rng(rng_)
  {}

  virtual ~weight_initializer() = default;

  virtual scalar operator()() const
  {
    return 0;
  }

  virtual void initialize_weights(eigen::matrix& W)
  {
    initialize_matrix(W, *this);
  }

  virtual void initialize_weights(mkl::sparse_matrix_csr<scalar>& W)
  {
    initialize_matrix(W, *this);
  }

  virtual void initialize_bias(eigen::vector& b)
  {
    b.array() = scalar(0);
  }
};

struct uniform_weight_initializer: public weight_initializer
{
  scalar low;
  scalar high;

  explicit uniform_weight_initializer(std::mt19937& rng, scalar low_ = -1.0, scalar high_ = 1.0)
  : weight_initializer(rng), low(low_), high(high_)
  {}

  scalar operator()() const override
  {
    std::uniform_real_distribution<scalar> dist(low, high);
    return dist(rng);
  }

  void initialize_weights(eigen::matrix& W) override  // TODO: avoid the need for overriding this method
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override  // TODO: avoid the need for overriding this method
  {
    initialize_matrix(W, *this);
  }
};

struct xavier_weight_initializer: public weight_initializer
{
  scalar x;

  xavier_weight_initializer(std::mt19937& rng, long columns)
   : weight_initializer(rng)
  {
    x = scalar(1.0) / std::sqrt(scalar(columns));
  }

  scalar operator()() const override
  {
    std::uniform_real_distribution<scalar> dist(-x, x);
    return dist(rng);
  }

  void initialize_weights(eigen::matrix& W) override
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override
  {
    initialize_matrix(W, *this);
  }
};

struct xavier_normalized_weight_initializer: public weight_initializer
{
  scalar x;

  xavier_normalized_weight_initializer(std::mt19937& rng, long rows, long columns)
   : weight_initializer(rng)
  {
    x = std::sqrt(scalar(6.0)) / std::sqrt(scalar(rows + columns));
  }

  scalar operator()() const override
  {
    std::uniform_real_distribution<scalar> dist(-x, x);
    return dist(rng);
  }

  void initialize_weights(eigen::matrix& W) override
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override
  {
    initialize_matrix(W, *this);
  }
};

struct he_weight_initializer: public weight_initializer
{
  scalar mean;
  scalar std;

  he_weight_initializer(std::mt19937& rng, long columns)
   : weight_initializer(rng)
  {
    mean = scalar(0);
    std = std::sqrt(scalar(2) / scalar(columns));
  }

  scalar operator()() const override
  {
    std::normal_distribution<scalar> dist(mean, std);
    return dist(rng);
  }

  void initialize_weights(eigen::matrix& W) override
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override
  {
    initialize_matrix(W, *this);
  }
};

struct zero_weight_initializer: public weight_initializer
{
  explicit zero_weight_initializer(std::mt19937& rng)
    : weight_initializer(rng)
  {}

  scalar operator()() const override
  {
    return scalar(0);
  }

  void initialize_weights(eigen::matrix& W) override
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override
  {
    initialize_matrix(W, *this);
  }
};

// used for testing
struct ten_weight_initializer: public weight_initializer
{
  explicit ten_weight_initializer(std::mt19937& rng)
    : weight_initializer(rng)
  {}

  scalar operator()() const override
  {
    return scalar(10);
  }

  void initialize_weights(eigen::matrix& W) override
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override
  {
    initialize_matrix(W, *this);
  }
};

struct pytorch_weight_initializer: public weight_initializer
{
  scalar x;

  pytorch_weight_initializer(std::mt19937& rng, long rows, long columns)
    : weight_initializer(rng)
  {
    x = std::sqrt(scalar(6.0)) / std::sqrt(scalar(rows + columns));
  }

  scalar operator()() const override
  {
    std::uniform_real_distribution<scalar> dist(-x, x);
    return dist(rng);
  }

  void initialize_bias(eigen::vector& b) override
  {
    b.array() = scalar(0.01);  // initialize b with small positive values
  }

  void initialize_weights(eigen::matrix& W) override
  {
    initialize_matrix(W, *this);
  }

  void initialize_weights(mkl::sparse_matrix_csr<scalar>& W) override
  {
    initialize_matrix(W, *this);
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
  tensorflow,
  zero,
  ten  // for testing
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
    case weight_initialization::zero: out << "zero"; break;
    case weight_initialization::ten: out << "ten"; break;
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
  else if (text == "zero")
  {
    return weight_initialization::zero;
  }
  else if (text == "ten")
  {
    return weight_initialization::ten;
  }
  throw std::runtime_error("could not parse weight initialization '" + text + "'");
}

template <typename Matrix>
std::shared_ptr<weight_initializer> make_weight_initializer(weight_initialization w, Matrix& W, std::mt19937& rng)
{
  switch(w)
  {
    case weight_initialization::he: return std::make_shared<he_weight_initializer>(rng, W.cols());
    case weight_initialization::xavier: return std::make_shared<xavier_weight_initializer>(rng, W.cols());
    case weight_initialization::xavier_normalized: return std::make_shared<xavier_normalized_weight_initializer>(rng, W.rows(), W.cols());
    case weight_initialization::pytorch: return std::make_shared<pytorch_weight_initializer>(rng, W.rows(), W.cols());
    case weight_initialization::default_: // TODO: implement this
    case weight_initialization::tensorflow: // TODO: implement this
    case weight_initialization::uniform: return std::make_shared<uniform_weight_initializer>(rng);
    case weight_initialization::zero: return std::make_shared<zero_weight_initializer>(rng);
    case weight_initialization::ten: return std::make_shared<ten_weight_initializer>(rng);
  }
  throw std::runtime_error("make_weight_initializer: unsupported weight initialization " + std::to_string(static_cast<int>(w)));
}

template <typename Matrix>
void initialize_weights(weight_initialization w, Matrix& W, eigen::vector& b, std::mt19937& rng)
{
  auto init = make_weight_initializer(w, W, rng);
  init->initialize_weights(W);
  init->initialize_bias(b);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_WEIGHTS_H
