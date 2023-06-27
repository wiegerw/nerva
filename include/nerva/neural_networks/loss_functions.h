// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/loss_functions.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_H
#define NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_H

#include "nerva/neural_networks/matrix_operations.h"
#include <cmath>

namespace nerva {

template <typename Vector>
auto squared_error_loss_colwise(const Vector& y, const Vector& t)
{
  return dot(y - t, y - t);
}

template <typename Vector>
auto squared_error_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  return 2 * (y - t);
}

template <typename Matrix>
auto Squared_error_loss_colwise(const Matrix& Y, const Matrix& T)
{
  return elements_sum(hadamard(Y - T, Y - T));
}

template <typename Matrix>
auto Squared_error_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  return 2 * (Y - T);
}

template <typename Vector>
auto mean_squared_error_loss_colwise(const Vector& y, const Vector& t)
{
  auto K = y.size();
  return squared_error_loss_colwise(y, t) / K;
}

template <typename Vector>
auto mean_squared_error_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  auto K = y.size();
  return squared_error_loss_colwise_gradient(y, t) / K;
}

template <typename Matrix>
auto Mean_squared_error_loss_colwise(const Matrix& Y, const Matrix& T)
{
  auto K = Y.cols();
  auto N = Y.rows();
  return Squared_error_loss_colwise(Y, T) / (K * N);
}

template <typename Matrix>
auto Mean_squared_error_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.cols();
  auto N = Y.rows();
  return Squared_error_loss_colwise_gradient(Y, T) / (K * N);
}

template <typename Vector>
auto cross_entropy_loss_colwise(const Vector& y, const Vector& t)
{
  return -dot(t, log(y));
}

template <typename Vector>
auto cross_entropy_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  return -hadamard(t, inverse(y));
}

template <typename Matrix>
auto Cross_entropy_loss_colwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, log(Y)));
}

template <typename Matrix>
auto Cross_entropy_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  return -hadamard(T, inverse(Y));
}

template <typename Vector>
auto softmax_cross_entropy_loss_colwise(const Vector& y, const Vector& t)
{
  return -dot(t, log_softmax_colwise(y));
}

template <typename Vector>
auto softmax_cross_entropy_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  return elements_sum(t) * softmax_colwise(y) - t;
}

template <typename Vector>
auto softmax_cross_entropy_loss_colwise_gradient_one_hot(const Vector& y, const Vector& t)
{
  return softmax_colwise(y) - t;
}

template <typename Matrix>
auto Softmax_cross_entropy_loss_colwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, log_softmax_colwise(Y)));
}

template <typename Matrix>
auto Softmax_cross_entropy_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.cols();
  return hadamard(softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T;
}

template <typename Matrix>
auto Softmax_cross_entropy_loss_colwise_gradient_one_hot(const Matrix& Y, const Matrix& T)
{
  return softmax_colwise(Y) - T;
}

template <typename Vector>
auto stable_softmax_cross_entropy_loss_colwise(const Vector& y, const Vector& t)
{
  return -dot(t, stable_log_softmax_colwise(y));
}

template <typename Vector>
auto stable_softmax_cross_entropy_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  return elements_sum(t) * stable_softmax_colwise(y) - t;
}

template <typename Vector>
auto stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(const Vector& y, const Vector& t)
{
  return stable_softmax_colwise(y) - t;
}

template <typename Matrix>
auto Stable_softmax_cross_entropy_loss_colwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, stable_log_softmax_colwise(Y)));
}

template <typename Matrix>
auto Stable_softmax_cross_entropy_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.cols();
  return hadamard(stable_softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T;
}

template <typename Matrix>
auto Stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(const Matrix& Y, const Matrix& T)
{
  return stable_softmax_colwise(Y) - T;
}

template <typename Vector>
auto logistic_cross_entropy_loss_colwise(const Vector& y, const Vector& t)
{
  return -dot(t, log(Sigmoid(y)));
}

template <typename Vector>
auto logistic_cross_entropy_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  return hadamard(t, Sigmoid(y)) - t;
}

template <typename Matrix>
auto Logistic_cross_entropy_loss_colwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, log(Sigmoid(Y))));
}

template <typename Matrix>
auto Logistic_cross_entropy_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  return hadamard(T, Sigmoid(Y)) - T;
}

template <typename Vector>
auto negative_log_likelihood_loss_colwise(const Vector& y, const Vector& t)
{
  return -std::log(dot(y, t));
}

template <typename Vector>
auto negative_log_likelihood_loss_colwise_gradient(const Vector& y, const Vector& t)
{
  return (-1 / dot(y, t)) * t;
}

template <typename Matrix>
auto Negative_log_likelihood_loss_colwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(log(columns_sum(hadamard(Y, T))));
}

template <typename Matrix>
auto Negative_log_likelihood_loss_colwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.cols();
  return -hadamard(row_repeat(inverse(columns_sum(hadamard(Y, T))), K), T);
}

template <typename Vector>
auto squared_error_loss_rowwise(const Vector& y, const Vector& t)
{
  return dot(y - t, y - t);
}

template <typename Vector>
auto squared_error_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  return 2 * (y - t);
}

template <typename Matrix>
auto Squared_error_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  return elements_sum(hadamard(Y - T, Y - T));
}

template <typename Matrix>
auto Squared_error_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  return 2 * (Y - T);
}

template <typename Vector>
auto mean_squared_error_loss_rowwise(const Vector& y, const Vector& t)
{
  auto K = y.size();
  return squared_error_loss_rowwise(y, t) / K;
}

template <typename Vector>
auto mean_squared_error_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  auto K = y.size();
  return squared_error_loss_rowwise_gradient(y, t) / K;
}

template <typename Matrix>
auto Mean_squared_error_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  auto N = Y.cols();
  auto K = Y.rows();
  return Squared_error_loss_rowwise(Y, T) / (K * N);
}

template <typename Matrix>
auto Mean_squared_error_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto N = Y.cols();
  auto K = Y.rows();
  return Squared_error_loss_rowwise_gradient(Y, T) / (K * N);
}

template <typename Vector>
auto cross_entropy_loss_rowwise(const Vector& y, const Vector& t)
{
  return -dot(t, log(y));
}

template <typename Vector>
auto cross_entropy_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  return -hadamard(t, inverse(y));
}

template <typename Matrix>
auto Cross_entropy_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, log(Y)));
}

template <typename Matrix>
auto Cross_entropy_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  return -hadamard(T, inverse(Y));
}

template <typename Vector>
auto softmax_cross_entropy_loss_rowwise(const Vector& y, const Vector& t)
{
  return -dot(t, log_softmax_rowwise(y));
}

template <typename Vector>
auto softmax_cross_entropy_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  return softmax_rowwise(y) * elements_sum(t) - t;
}

template <typename Vector>
auto softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Vector& y, const Vector& t)
{
  return softmax_rowwise(y) - t;
}

template <typename Matrix>
auto Softmax_cross_entropy_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, log_softmax_rowwise(Y)));
}

template <typename Matrix>
auto Softmax_cross_entropy_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.rows();
  return hadamard(softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T;
}

template <typename Matrix>
auto Softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Matrix& Y, const Matrix& T)
{
  return softmax_rowwise(Y) - T;
}

template <typename Vector>
auto stable_softmax_cross_entropy_loss_rowwise(const Vector& y, const Vector& t)
{
  return -dot(t, stable_log_softmax_rowwise(y));
}

template <typename Vector>
auto stable_softmax_cross_entropy_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  return stable_softmax_rowwise(y) * elements_sum(t) - t;
}

template <typename Vector>
auto stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Vector& y, const Vector& t)
{
  return stable_softmax_rowwise(y) - t;
}

template <typename Matrix>
auto Stable_softmax_cross_entropy_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, stable_log_softmax_rowwise(Y)));
}

template <typename Matrix>
auto Stable_softmax_cross_entropy_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.rows();
  return hadamard(stable_softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T;
}

template <typename Matrix>
auto Stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(const Matrix& Y, const Matrix& T)
{
  return stable_softmax_rowwise(Y) - T;
}

template <typename Vector>
auto logistic_cross_entropy_loss_rowwise(const Vector& y, const Vector& t)
{
  return -dot(t, log(Sigmoid(y)));
}

template <typename Vector>
auto logistic_cross_entropy_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  return hadamard(t, Sigmoid(y)) - t;
}

template <typename Matrix>
auto Logistic_cross_entropy_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(hadamard(T, log(Sigmoid(Y))));
}

template <typename Matrix>
auto Logistic_cross_entropy_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  return hadamard(T, Sigmoid(Y)) - T;
}

template <typename Vector>
auto negative_log_likelihood_loss_rowwise(const Vector& y, const Vector& t)
{
  return -std::log(dot(y, t));
}

template <typename Vector>
auto negative_log_likelihood_loss_rowwise_gradient(const Vector& y, const Vector& t)
{
  return (-1 / dot(y, t)) * t;
}

template <typename Matrix>
auto Negative_log_likelihood_loss_rowwise(const Matrix& Y, const Matrix& T)
{
  return -elements_sum(log(rows_sum(hadamard(Y, T))));
}

template <typename Matrix>
auto Negative_log_likelihood_loss_rowwise_gradient(const Matrix& Y, const Matrix& T)
{
  auto K = Y.rows();
  return -hadamard(column_repeat(inverse(rows_sum(hadamard(Y, T))), K), T);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LOSS_FUNCTIONS_H
