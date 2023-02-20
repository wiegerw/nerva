// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/optimizers.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_OPTIMIZERS_H
#define NERVA_NEURAL_NETWORKS_OPTIMIZERS_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/matrix.h"
#include "nerva/neural_networks/mkl_matrix.h"
#include "nerva/utilities/parse_numbers.h"
#include "fmt/format.h"

namespace nerva {

struct layer_optimizer
{
  virtual void update(scalar eta) = 0;

  [[nodiscard]] virtual std::string to_string() const = 0;

  // If the layer is sparse, the nonzero entries of the weight matrices in the
  // optimizer need to be updated
  virtual void reset_stencil(const mkl::sparse_matrix_csr<scalar>& W)
  { }

  virtual ~layer_optimizer() = default;
};

template <typename Matrix>
struct gradient_descent_optimizer: public layer_optimizer
{
  Matrix& W;
  Matrix& DW;
  eigen::vector& b;
  eigen::vector& Db;

  gradient_descent_optimizer(Matrix& W_, Matrix& DW_, eigen::vector& b_, eigen::vector& Db_)
    : W(W_), DW(DW_), b(b_), Db(Db_)
  {}

  [[nodiscard]] std::string to_string() const override
  {
    return "GradientDescent()";
  }

  void update(scalar eta) override
  {
    if constexpr (std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value)
    {
      mkl::assign_matrix_sum(W, DW, scalar(1), -eta);
    }
    else
    {
      W -= eta * DW;
    }
    b -= eta * Db;
  }
};

template <typename Matrix>
struct momentum_optimizer: public gradient_descent_optimizer<Matrix>
{
  using super = gradient_descent_optimizer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  Matrix delta_W;
  eigen::vector delta_b;
  scalar mu;

  momentum_optimizer(Matrix& W, Matrix& DW, eigen::vector& b, eigen::vector& Db, scalar mu_)
   : super(W, DW, b, Db),
     delta_W(W.rows(), W.cols()),
     delta_b(b.size()),
     mu(mu_)
  {
    if constexpr (std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value)
    {
      delta_W.assign(W, scalar(0)); // copy the stencil of W and fill the matrix with zeroes
    }
    else
    {
      initialize_matrix(delta_W, scalar(0));
    }
    delta_b.array() = scalar(0);
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Momentum({:7.5f})", mu);
  }

  void update(scalar eta) override
  {
    if constexpr (std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value)
    {
      mkl::assign_matrix_sum(delta_W, DW, mu, -eta);
      mkl::assign_matrix_sum(W, delta_W, scalar(1), scalar(1));
    }
    else
    {
      delta_W = mu * delta_W - eta * DW;
      W += delta_W;
    }

    delta_b = mu * delta_b - eta * Db;
    b += delta_b;
  }

  void reset_stencil(const mkl::sparse_matrix_csr<scalar>& W) override
  {
    if constexpr (IsSparse)
    {
      delta_W = W;
      delta_W = scalar(0);
    }
  }
};

template <typename Matrix>
struct nesterov_optimizer: public gradient_descent_optimizer<Matrix>
{
  using super = gradient_descent_optimizer<Matrix>;
  using super::W;
  using super::DW;
  using super::b;
  using super::Db;
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  Matrix delta_W;
  Matrix delta_W_prev;
  eigen::vector delta_b;
  eigen::vector delta_b_prev;
  scalar mu;

  nesterov_optimizer(Matrix& W, Matrix& DW, eigen::vector& b, eigen::vector& Db, scalar mu_)
      : super(W, DW, b, Db),
        delta_W(W.rows(), W.cols()),
        delta_W_prev(W.rows(), W.cols()),
        delta_b(b.size()),
        delta_b_prev(b.size()),
        mu(mu_)
  {
    if constexpr (std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value)
    {
      delta_W.assign(W, scalar(0)); // copy the stencil of W and fill the matrix with zeroes
      delta_W_prev.assign(W, scalar(0)); // copy the stencil of W and fill the matrix with zeroes
    }
    else
    {
      delta_W.array() = scalar(0);
      delta_W_prev.array() = scalar(0);
    }
    delta_b.array() = scalar(0);
    delta_b_prev.array() = scalar(0);
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("Nesterov({:7.5f})", mu);
  }

  void update(scalar eta) override
  {
    if constexpr (std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value)
    {
      mkl::assign_matrix(delta_W_prev, delta_W);
      mkl::assign_matrix_sum(delta_W, DW, mu, -eta);
      mkl::assign_matrix_sum(W, delta_W, delta_W_prev, scalar(1), scalar(1) + mu, -mu);
    }
    else
    {
      delta_W_prev = delta_W;
      delta_W = mu * delta_W - eta * DW;
      W += (-mu * delta_W_prev + (scalar(1) + mu) * delta_W);
    }

    delta_b_prev = delta_b;
    delta_b = mu * delta_b - eta * b;
    b += (-mu * delta_b_prev + (scalar(1) + mu) * delta_b);
  }

  void reset_stencil(const mkl::sparse_matrix_csr<scalar>& W) override
  {
    if constexpr (IsSparse)
    {
      delta_W = W;
      delta_W = scalar(0);
      delta_W_prev = W;
      delta_W_prev = scalar(0);
    }
  }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_OPTIMIZERS_H
