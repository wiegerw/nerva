// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/optimizers.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/matrix.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/parse_numbers.h"
#include "fmt/format.h"

namespace nerva {

// Generic optimizer_function for dense or sparse matrices.
struct optimizer_function
{
  virtual void update(scalar eta) = 0;

  // Update the support (i.e. the set of nonzero entries). Only applies to sparse matrices.
  virtual void reset_support()
  {}

  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  virtual ~optimizer_function() = default;
};

template <typename T>
struct gradient_descent_optimizer: public optimizer_function
{
  T& x;
  T& Dx;

  gradient_descent_optimizer(T& x_, T& Dx_)
    : x(x_), Dx(Dx_)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "gradient_descent";
  }

  void update(scalar eta) override
  {
    if constexpr (std::is_same<T, mkl::sparse_matrix_csr<scalar>>::value)
    {
      mkl::ss_sum(x, Dx, scalar(1), -eta);
    }
    else
    {
      x -= eta * Dx;
    }
  }
};

template <typename T>
struct momentum_optimizer: public gradient_descent_optimizer<T>
{
  using super = gradient_descent_optimizer<T>;
  using super::x;
  using super::Dx;
  using super::reset_support;
  static const bool IsSparse = std::is_same_v<T, mkl::sparse_matrix_csr<scalar>>;

  T delta_x;
  scalar mu;

  momentum_optimizer(T& x, T& Dx, scalar mu_)
    : super(x, Dx),
      delta_x(x.rows(), x.cols()),
      mu(mu_)
  {
    if constexpr (IsSparse)
    {
      reset_support();
    }
    else
    {
      delta_x.array() = scalar(0);
    }
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return fmt::format("Momentum({:7.5f})", mu);
  }

  void update(scalar eta) override
  {
    if constexpr (IsSparse)
    {
      mkl::ss_sum(delta_x, Dx, mu, -eta);
      mkl::ss_sum(x, delta_x, scalar(1), scalar(1));
    }
    else
    {
      delta_x = mu * delta_x - eta * Dx;
      x += delta_x;
    }
  }

  void reset_support() override
  {
    if constexpr (IsSparse)
    {
      delta_x.reset_support(x);
    }
  }
};

template <typename T>
struct nesterov_optimizer: public gradient_descent_optimizer<T>
{
  using super = gradient_descent_optimizer<T>;
  using super::x;
  using super::Dx;
  static const bool IsSparse = std::is_same_v<T, mkl::sparse_matrix_csr<scalar>>;

  T delta_x;
  T delta_x_prev;
  scalar mu;

  void reset_support() override
  {
    if constexpr (IsSparse)
    {
      delta_x.reset_support(x);
      delta_x_prev.reset_support(x);
    }
  }

  nesterov_optimizer(T& x, T& Dx, scalar mu_)
    : super(x, Dx),
      delta_x(x.rows(), x.cols()),
      delta_x_prev(x.rows(), x.cols()),
      mu(mu_)
  {
    if constexpr (IsSparse)
    {
      reset_support();
    }
    else
    {
      delta_x.array() = scalar(0);
      delta_x_prev.array() = scalar(0);
    }
  }

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return fmt::format("Nesterov({:7.5f})", mu);
  }

  void update(scalar eta) override
  {
    if constexpr (IsSparse)
    {
      mkl::assign_matrix(delta_x_prev, delta_x);
      mkl::ss_sum(delta_x, Dx, mu, -eta);
      mkl::sss_sum(x, delta_x, delta_x_prev, scalar(1), scalar(1) + mu, -mu);
    }
    else
    {
      delta_x_prev = delta_x;
      delta_x = mu * delta_x - eta * Dx;
      x += (-mu * delta_x_prev + (scalar(1) + mu) * delta_x);
    }
  }
};

struct composite_optimizer: public optimizer_function
{
  std::vector<std::shared_ptr<optimizer_function>> optimizers;

  composite_optimizer() = default;

  composite_optimizer(const composite_optimizer& other)
    : optimizers(other.optimizers)
  {}

  composite_optimizer(std::initializer_list<std::shared_ptr<optimizer_function>> items)
    : optimizers(items)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return optimizers.front()->to_string();
  }

  void update(scalar eta) override
  {
    for (auto& optimizer: optimizers)
    {
      optimizer->update(eta);
    }
  }

  void reset_support() override
  {
    for (auto& optimizer: optimizers)
    {
      optimizer->reset_support();
    }
  }
};

template <typename... Args>
auto make_composite_optimizer(Args&&... args) -> std::shared_ptr<composite_optimizer>
{
  return std::make_shared<composite_optimizer>(std::initializer_list<std::shared_ptr<optimizer_function>>{std::forward<Args>(args)...});
}

template <typename T>
auto parse_optimizer(const std::string& text, T& x, T& Dx) -> std::shared_ptr<optimizer_function>
{
  auto func = utilities::parse_function_call(text);

  if (func.name == "GradientDescent")
  {
    return std::make_shared<gradient_descent_optimizer<T>>(x, Dx);
  }
  else if (func.name == "Momentum")
  {
    scalar mu = func.as_scalar("momentum");
    return std::make_shared<momentum_optimizer<T>>(x, Dx, mu);
  }
  else if (func.name == "Nesterov")
  {
    scalar mu = func.as_scalar("momentum");
    return std::make_shared<nesterov_optimizer<T>>(x, Dx, mu);
  }
  else
  {
    throw std::runtime_error("unknown optimizer '" + text + "'");
  }
}

// TODO: remove the classes below; currently they are still used in the tests and the python bindings
template <typename Matrix>
struct gradient_descent_linear_layer_optimizer: public optimizer_function
{
  gradient_descent_optimizer<Matrix> optimizer_W;
  gradient_descent_optimizer<eigen::matrix> optimizer_b;

  gradient_descent_linear_layer_optimizer(Matrix& W, Matrix& DW, eigen::matrix& b, eigen::matrix& Db)
    : optimizer_W(W, DW), optimizer_b(b, Db)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return "GradientDescent()";
  }

  void update(scalar eta) override
  {
    optimizer_W.update(eta);
    optimizer_b.update(eta);
  }

  void reset_support() override
  {
    optimizer_W.reset_support();
    optimizer_b.reset_support();
  }
};

template <typename Matrix>
struct momentum_linear_layer_optimizer: public optimizer_function
{
  momentum_optimizer<Matrix> optimizer_W;
  momentum_optimizer<eigen::matrix> optimizer_b;

  momentum_linear_layer_optimizer(Matrix& W, Matrix& DW, eigen::matrix& b, eigen::matrix& Db, scalar mu)
    : optimizer_W(W, DW, mu), optimizer_b(b, Db, mu)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return fmt::format("Momentum({:7.5f})", optimizer_W.mu);
  }

  void update(scalar eta) override
  {
    optimizer_W.update(eta);
    optimizer_b.update(eta);
  }

  void reset_support() override
  {
    optimizer_W.reset_support();
    optimizer_b.reset_support();
  }
};

template <typename Matrix>
struct nesterov_linear_layer_optimizer: public optimizer_function
{
  nesterov_optimizer<Matrix> optimizer_W;
  nesterov_optimizer<eigen::matrix> optimizer_b;

  nesterov_linear_layer_optimizer(Matrix& W, Matrix& DW, eigen::matrix& b, eigen::matrix& Db, scalar mu)
    : optimizer_W(W, DW, mu), optimizer_b(b, Db, mu)
  {}

  [[nodiscard]] auto to_string() const -> std::string override
  {
    return fmt::format("Nesterov({:7.5f})", optimizer_W.mu);
  }

  void update(scalar eta) override
  {
    optimizer_W.update(eta);
    optimizer_b.update(eta);
  }

  void reset_support() override
  {
    optimizer_W.reset_support();
    optimizer_b.reset_support();
  }
};

} // namespace nerva

