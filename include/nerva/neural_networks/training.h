// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/training.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_TRAINING_H
#define NERVA_NEURAL_NETWORKS_TRAINING_H

#include "nerva/neural_networks/check_gradients.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/print.h"
#include "nerva/utilities/stopwatch.h"
#include "fmt/format.h"
#include <algorithm>
#include <iomanip>

namespace nerva {

template <typename EigenMatrix>
double compute_accuracy(multilayer_perceptron& M, const EigenMatrix& Xtest, const EigenMatrix& Ttest, long Q)
{
  auto is_correct = [](const eigen::vector& y, const eigen::vector& t)
  {
    auto i = std::max_element(y.begin(), y.end()) - y.begin(); // i is the index of the largest element
    return t[i] == 1;
  };

  long N = Xtest.cols(); // the number of examples
  long L = Ttest.rows(); // the number of outputs
  auto K = N / Q;        // the number of batches
  eigen::matrix Ybatch(L, Q);
  std::size_t total_correct = 0;

  for (long k = 0; k < K; k++)
  {
    auto batch = Eigen::seqN(k * Q, Q);
    auto Xbatch = Xtest(Eigen::all, batch);
    auto Tbatch = Ttest(Eigen::all, batch);
    M.feedforward(Xbatch, Ybatch);
    for (long i = 0; i < Q; i++)
    {
      const auto& y = Ybatch.col(i);
      const auto& t = Tbatch.col(i);
      if (is_correct(y, t))
      {
        total_correct++;
      }
    }
  }
  return static_cast<double>(total_correct) / N;
}

inline
double compute_loss(multilayer_perceptron& M, const std::shared_ptr<loss_function>& loss, const eigen::matrix& X, const eigen::matrix& T, long Q)
{
  long N = X.cols(); // the number of examples
  long L = T.rows(); // the number of outputs
  auto K = N / Q;    // the number of batches
  double total_loss = 0.0;
  eigen::matrix Ybatch(L, Q);

  for (long k = 0; k < K; k++)
  {
    auto batch = Eigen::seqN(k * Q, Q);
    auto Xbatch = X(Eigen::all, batch);
    auto Tbatch = T(Eigen::all, batch);
    M.feedforward(Xbatch, Ybatch);
    total_loss += loss->value(Ybatch, Tbatch);
  }
  return total_loss / N; // return the average loss
}

template <typename DataSet>
void compute_statistics(multilayer_perceptron& M,
                        scalar lr,
                        const std::shared_ptr<loss_function>& loss,
                        const DataSet& data,
                        long Q, // the batch size
                        int epoch,
                        bool full_statistics,
                        double elapsed_seconds = -1.0
                       )
{
  std::cout << fmt::format("epoch {:3d}", epoch + 1);
  if (full_statistics)
  {
    auto training_loss = compute_loss(M, loss, data.Xtrain, data.Ttrain, Q);
    auto training_accuracy = compute_accuracy(M, data.Xtrain, data.Ttrain, Q);
    auto test_accuracy = compute_accuracy(M, data.Xtest, data.Ttest, Q);
    std::cout << fmt::format(" lr: {:.8f}  loss: {:.8f}  train accuracy: {:.8f}  test accuracy: {:.8f}", lr, training_loss, training_accuracy, test_accuracy);
  }
  if (elapsed_seconds >= 0)
  {
    std::cout << fmt::format(" time: {:.8f}s", elapsed_seconds);
  }
  std::cout << std::endl;
}

// Returns the test accuracy and the total training time
template <typename DataSet, typename RandomNumberGenerator>
std::pair<double, double> stochastic_gradient_descent(
  multilayer_perceptron& M,
  const std::shared_ptr<loss_function>& loss,
  DataSet& data,
  const sgd_options& options,
  const std::shared_ptr<learning_rate_scheduler>& learning_rate,
  RandomNumberGenerator rng)
{
  double total_training_time = 0;
  long N = data.Xtrain.cols(); // the number of examples
  long L = data.Ttrain.rows(); // the number of outputs
  std::vector<long> I(N);
  std::iota(I.begin(), I.end(), 0);
  eigen::matrix Y(L, options.batch_size);
  long K = N / options.batch_size; // the number of batches
  utilities::stopwatch watch;
  scalar eta = learning_rate->operator()(0);

  compute_statistics(M, eta, loss, data, options.batch_size, -1, options.statistics, 0.0);

  for (int epoch = 0; epoch < options.epochs; ++epoch)
  {
    watch.reset();

    if (epoch > 0 && options.regrow_rate > 0)
    {
      std::cout << "regrow" << std::endl;
      M.regrow(options.regrow_rate, weight_initialization::xavier, options.regrow_separate_positive_negative, rng);
    }

    if (options.shuffle)
    {
      std::shuffle(I.begin(), I.end(), rng);      // shuffle the examples at the start of each epoch
    }
    M.renew_dropout_mask(rng);
    eta = learning_rate->operator()(epoch);       // update the learning at the start of each epoch
    eigen::matrix DY(L, options.batch_size);

    for (long k = 0; k < K; k++)
    {
      eigen::eigen_slice batch(I.begin() + k * options.batch_size, options.batch_size);
      auto X = data.Xtrain(Eigen::all, batch);
      auto T = data.Ttrain(Eigen::all, batch);
      M.feedforward(X, Y);

      if (options.gradient_step > 0)
      {
        DY = loss->gradient(Y, T);
        auto f = [loss, &Y, &T]() { return loss->value(Y, T); };
        check_gradient("DY", f, Y, DY, options.gradient_step);
      }
      else
      {
        DY = loss->gradient(Y, T) / options.batch_size;  // pytorch does it like this
      }

      if (options.debug)
      {
        std::cout << "epoch: " << epoch << " batch: " << k << std::endl;
        // print_model_info(M);
        eigen::print_numpy_matrix("X", X.transpose());
        eigen::print_numpy_matrix("Y", Y.transpose());
        eigen::print_numpy_matrix("DY", DY.transpose());
      }

      M.backpropagate(Y, DY);

      if (options.gradient_step > 0)
      {
        M.check_gradients(loss, T, options.gradient_step);
      }

      M.optimize(eta);
    }
    double seconds = watch.seconds();
    total_training_time += seconds;
    compute_statistics(M, eta, loss, data, options.batch_size, epoch, options.statistics, seconds);
  }
  double test_accuracy = compute_accuracy(M, data.Xtest, data.Ttest, options.batch_size);
  std::cout << fmt::format("Accuracy of the network on the {} test examples: {:.2f}%", data.Xtest.cols(), test_accuracy * 100.0) << std::endl;
  std::cout << fmt::format("Total training time for the {} epochs: {:.8f}s\n", options.epochs, total_training_time);
  return {test_accuracy, total_training_time};
}

// Loads a new dataset at every epoch from the directory datadir
template <typename RandomNumberGenerator>
std::pair<double, double> stochastic_gradient_descent_preprocessed(
  multilayer_perceptron& M,
  const std::shared_ptr<loss_function>& loss,
  const std::string& datadir,
  const sgd_options& options,
  const std::shared_ptr<learning_rate_scheduler>& learning_rate,
  RandomNumberGenerator rng)
{
  auto path = std::filesystem::path(datadir);
  datasets::dataset data;

  // read the first dataset
  data.load(path / "epoch0.npz");

  double total_training_time = 0;
  long N = data.Xtrain.cols(); // the number of examples
  long L = data.Ttrain.rows(); // the number of outputs
  std::vector<long> I(N);
  std::iota(I.begin(), I.end(), 0);
  eigen::matrix Y(L, options.batch_size);
  long K = N / options.batch_size; // the number of batches
  utilities::stopwatch watch;
  scalar eta = learning_rate->operator()(0);

  compute_statistics(M, eta, loss, data, options.batch_size, -1, options.statistics, 0.0);

  for (int epoch = 0; epoch < options.epochs; ++epoch)
  {
    // read the next dataset
    if (epoch > 0)
    {
      data.load((path / ("epoch" + std::to_string(epoch) + ".npz")).native());
    }
    watch.reset();

    if (epoch > 0 && options.regrow_rate > 0)
    {
      std::cout << "regrow" << std::endl;
      M.regrow(options.regrow_rate, weight_initialization::xavier, options.regrow_separate_positive_negative, rng);
    }

    if (options.shuffle)
    {
      std::shuffle(I.begin(), I.end(), rng);      // shuffle the examples at the start of each epoch
    }

    M.renew_dropout_mask(rng);
    eta = learning_rate->operator()(epoch);       // update the learning at the start of each epoch
    eigen::matrix DY(L, options.batch_size);

    for (long k = 0; k < K; k++)
    {
      eigen::eigen_slice batch(I.begin() + k * options.batch_size, options.batch_size);
      auto X = data.Xtrain(Eigen::all, batch);
      auto T = data.Ttrain(Eigen::all, batch);
      M.feedforward(X, Y);

      if (options.gradient_step > 0)
      {
        DY = loss->gradient(Y, T);
        auto f = [loss, &Y, &T]() { return loss->value(Y, T); };
        check_gradient("DY", f, Y, DY, options.gradient_step);
      }
      else
      {
        DY = loss->gradient(Y, T) / options.batch_size;  // pytorch does it like this
      }

      if (options.debug)
      {
        std::cout << "epoch: " << epoch << " batch: " << k << std::endl;
        // print_model_info(M);
        eigen::print_numpy_matrix("X", X.transpose());
        eigen::print_numpy_matrix("Y", Y.transpose());
        eigen::print_numpy_matrix("DY", DY.transpose());
      }

      M.backpropagate(Y, DY);

      if (options.gradient_step > 0)
      {
        M.check_gradients(loss, T, options.gradient_step);
      }

      M.optimize(eta);
    }
    double seconds = watch.seconds();
    total_training_time += seconds;
    compute_statistics(M, eta, loss, data, options.batch_size, epoch, options.statistics, seconds);
  }
  double test_accuracy = compute_accuracy(M, data.Xtest, data.Ttest, options.batch_size);
  std::cout << fmt::format("Accuracy of the network on the {} test examples: {:.2f}%", data.Xtest.cols(), test_accuracy * 100.0) << std::endl;
  std::cout << fmt::format("Total training time for the {} epochs: {:.8f}s\n", options.epochs, total_training_time);
  return {test_accuracy, total_training_time};
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_TRAINING_H
