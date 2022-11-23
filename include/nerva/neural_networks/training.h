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

template <typename MultilayerPerceptron, typename EigenMatrix>
double compute_accuracy(MultilayerPerceptron& M, const EigenMatrix& Xtest, const EigenMatrix& Ttest)
{
  auto is_correct = [](const eigen::vector& y, const eigen::vector& t)
  {
    auto i = std::max_element(y.begin(), y.end()) - y.begin(); // i is the index of the largest element
    return t[i] == 1;
  };

  long N = Xtest.cols(); // the number of examples
  long L = Ttest.rows(); // the number of outputs
  std::size_t total_correct = 0;
  eigen::matrix y(L, 1);

  for (unsigned int i = 0; i < N; ++i)
  {
    const auto& t = Ttest.col(i);
    const auto& x = Xtest.col(i);
    M.feedforward(x, y);
    if (is_correct(y, t))
    {
      total_correct++;
    }
  }
  return static_cast<double>(total_correct) / N;
}

template <typename MultilayerPerceptron, typename EigenMatrix>
double compute_accuracy_batch(MultilayerPerceptron& M, const EigenMatrix& Xtest, const EigenMatrix& Ttest, long Q)
{
  auto is_correct = [](const eigen::vector& y, const eigen::vector& t)
  {
    auto i = std::max_element(y.begin(), y.end()) - y.begin(); // i is the index of the largest element
    return t[i] == 1;
  };

  long N = Xtest.cols(); // the number of examples
  long L = Ttest.rows(); // the number of outputs
  auto K = N / Q;  // the number of batches
  eigen::matrix Ybatch(L, Q);
  std::size_t total_correct = 0;

  for (long k = 0; k < K; k++)
  {
    auto batch = Eigen::seqN(k * Q, Q);
    auto Xbatch = Xtest(Eigen::placeholders::all, batch);
    auto Tbatch = Ttest(Eigen::placeholders::all, batch);
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

template <typename MultilayerPerceptron>
double compute_loss(MultilayerPerceptron& M, const std::shared_ptr<loss_function>& loss, const eigen::matrix& X, const eigen::matrix& T)
{
  long N = X.cols(); // the number of examples
  long L = T.rows(); // the number of outputs
  double total_loss = 0.0;
  eigen::matrix y(L, 1);

  for (unsigned int i = 0; i < N; ++i)
  {
    const eigen::vector& t = T.col(i);
    const eigen::vector& x = X.col(i);
    M.feedforward(x, y);
    total_loss += loss->value(y, t);
  }
  return total_loss / N; // display the average loss
}

template <typename MultilayerPerceptron>
double compute_loss_batch(MultilayerPerceptron& M, const std::shared_ptr<loss_function>& loss, const eigen::matrix& X, const eigen::matrix& T, long Q)
{
  long N = X.cols(); // the number of examples
  long L = T.rows(); // the number of outputs
  auto K = N / Q;                // the number of batches
  double total_loss = 0.0;
  eigen::matrix Ybatch(L, Q);

  for (long k = 0; k < K; k++)
  {
    auto batch = Eigen::seqN(k * Q, Q);
    auto Xbatch = X(Eigen::placeholders::all, batch);
    auto Tbatch = T(Eigen::placeholders::all, batch);
    M.feedforward(Xbatch, Ybatch);
    total_loss += loss->value(Ybatch, Tbatch);
  }
  return total_loss / N; // display the average loss
}

template <typename MultilayerPerceptron, typename DataSet>
void compute_statistics(MultilayerPerceptron& M,
                        const std::shared_ptr<loss_function>& loss,
                        const DataSet& data,
                        unsigned int epoch,
                        bool full_statistics,
                        scalar elapsed_seconds = -1.0
                        )
{
  std::cout << fmt::format("epoch {:3d}", epoch + 1);
  if (full_statistics)
  {
    auto training_loss = compute_loss(M, loss, data.Xtrain, data.Ttrain);
    auto training_accuracy = compute_accuracy(M, data.Xtrain, data.Ttrain);
    auto test_accuracy = compute_accuracy(M, data.Xtest, data.Ttest);
    std::cout << fmt::format("  loss: {:7.4f}  train accuracy: {:7.4f}  test accuracy: {:7.4f}", training_loss, training_accuracy, test_accuracy);
  }
  if (elapsed_seconds >= 0)
  {
    std::cout << fmt::format("  time: {:7.4f}s", elapsed_seconds);
  }
  std::cout << std::endl;
}

template <typename MultilayerPerceptron, typename DataSet>
void compute_statistics_batch(MultilayerPerceptron& M,
                              const std::shared_ptr<loss_function>& loss,
                              const DataSet& data,
                              long Q, // the batch size
                              int epoch,
                              bool full_statistics,
                              scalar elapsed_seconds = -1.0
                              )
{
  std::cout << fmt::format("epoch {:3d}", epoch + 1);
  if (full_statistics)
  {
    auto training_loss = compute_loss_batch(M, loss, data.Xtrain, data.Ttrain, Q);
    auto training_accuracy = compute_accuracy_batch(M, data.Xtrain, data.Ttrain, Q);
    auto test_accuracy = compute_accuracy_batch(M, data.Xtest, data.Ttest, Q);
    std::cout << fmt::format(" loss: {:7.4f}  train accuracy: {:7.4f}  test accuracy: {:7.4f}", training_loss, training_accuracy, test_accuracy);
  }
  if (elapsed_seconds >= 0)
  {
    std::cout << fmt::format(" time: {:7.4f}s", elapsed_seconds);
  }
  std::cout << std::endl;
}

template <typename MultilayerPerceptron, typename DataSet, typename RandomNumberGenerator>
void stochastic_gradient_descent(
         MultilayerPerceptron& M,
         const std::shared_ptr<loss_function>& loss,
         const DataSet& data,
         const sgd_options& options,
         const std::shared_ptr<learning_rate_scheduler>& learning_rate,
         RandomNumberGenerator rng
  )
{
  std::cout << std::setprecision(4);
  long N = data.Xtrain.cols(); // the number of examples
  long D = data.Xtrain.rows(); // the number of features
  std::vector<long> I(N);
  std::iota(I.begin(), I.end(), 0);
  eigen::matrix y(D, 1);
  eigen::matrix dy(D, 1);
  utilities::stopwatch watch;
  compute_statistics(M, loss, data, -1, options.statistics, 0.0);

  for (unsigned int epoch = 0; epoch < options.epochs; ++epoch)
  {
    watch.reset();
    if (options.shuffle)
    {
      std::shuffle(I.begin(), I.end(), rng);   // shuffle the examples at the start of each epoch
    }
    M.renew_dropout_mask(rng);
    scalar eta = learning_rate->operator()(epoch);       // update the learning at the start of each epoch

    for (auto i: I)
    {
      NERVA_LOG(log::debug) << "--- sample " << i << std::endl;
      const auto& x = data.Xtrain.col(i);
      const auto& t = data.Ttrain.col(i);
      M.feedforward(x, y);
      dy = loss->gradient(y, t);
      if (options.check_gradients)
      {
        auto f = [&loss, &y, &t]() { return loss->value(y, t); };
        check_gradient("dy", f, y, dy, options.check_gradients_step);
      }
      M.backpropagate(y, dy);
      if (options.check_gradients)
      {
        M.check_gradients(loss, t, options.check_gradients_step);
      }
      M.optimize(eta);
    }

    compute_statistics(M, loss, data, epoch, options.statistics, watch.seconds());
  }
  std::cout << "Accuracy of the network on the " << data.Xtest.cols() << " test images: " << std::setw(10) << (100.0 * compute_accuracy(M, data.Xtest, data.Ttest)) << "%" << std::endl;
}

template <typename MultilayerPerceptron, typename DataSet, typename RandomNumberGenerator>
void minibatch_stochastic_gradient_descent(
  MultilayerPerceptron& M,
  const std::shared_ptr<loss_function>& loss,
  const DataSet& data,
  const sgd_options& options,
  const std::shared_ptr<learning_rate_scheduler>& learning_rate,
  RandomNumberGenerator rng
  )
{
  std::cout << std::setprecision(4);
  long N = data.Xtrain.cols(); // the number of examples
  long L = data.Ttrain.rows(); // the number of outputs

  std::vector<long> I(N);
  std::iota(I.begin(), I.end(), 0);
  eigen::matrix y(L, options.batch_size);
  eigen::matrix dy(L, options.batch_size);
  long K = N / options.batch_size; // the number of batches
  utilities::stopwatch watch;
  compute_statistics(M, loss, data, -1, options.statistics, 0.0);

  typename MultilayerPerceptron::gradient_type dWdb(M);

  for (unsigned int epoch = 0; epoch < options.epochs; ++epoch)
  {
    watch.reset();
    if (options.shuffle)
    {
      std::shuffle(I.begin(), I.end(), rng);// shuffle the examples at the start of each epoch
    }
    M.renew_dropout_mask(rng);
    scalar eta = learning_rate->operator()(epoch);       // update the learning at the start of each epoch

    for (long k = 0; k < K; k++)
    {
      dWdb.wipe();
      for (long i = k * options.batch_size; i < k * options.batch_size + options.batch_size; i++)
      {
        const auto& x = data.Xtrain.col(I[i]);
        const auto& t = data.Ttrain.col(I[i]);
        M.feedforward(x, y);
        dy = loss->gradient(y, t);
        M.backpropagate(y, dy);
        if (options.check_gradients)
        {
          M.check_gradients(loss, t, options.check_gradients_step);
        }
        dWdb.add(M);
      }
      dWdb.set(M);
      M.optimize(eta);
    }

    if (options.statistics)
    {
      compute_statistics(M, loss, data, epoch, options.statistics, watch.seconds());
    }
  }
  std::cout << "Accuracy of the network on the " << data.Xtest.cols() << " test images: " << std::setw(10) << (100.0 * compute_accuracy(M, data.Xtest, data.Ttest)) << "%" << std::endl;
}

// Returns the test accuracy and the total training time
template <typename MultilayerPerceptron, typename DataSet, typename RandomNumberGenerator>
std::pair<double, double> minibatch_gradient_descent(
  MultilayerPerceptron& M,
  const std::shared_ptr<loss_function>& loss,
  const DataSet& data,
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
  compute_statistics_batch(M, loss, data, options.batch_size, -1, options.statistics, 0.0);

  for (unsigned int epoch = 0; epoch < options.epochs; ++epoch)
  {
    watch.reset();
    if (options.shuffle)
    {
      std::shuffle(I.begin(), I.end(), rng);// shuffle the examples at the start of each epoch
    }
    M.renew_dropout_mask(rng);
    scalar eta = learning_rate->operator()(epoch);       // update the learning at the start of each epoch
    eigen::matrix DY(L, options.batch_size);

    for (long k = 0; k < K; k++)
    {
      eigen::eigen_slice batch(I.begin() + k * options.batch_size, options.batch_size);
      auto X = data.Xtrain(Eigen::placeholders::all, batch);
      auto T = data.Ttrain(Eigen::placeholders::all, batch);
      M.feedforward(X, Y);
      if (options.debug)
      {
        std::cout << "epoch: " << epoch << " batch: " << k << std::endl;
        eigen::print_numpy_matrix("X", X.transpose());
        eigen::print_numpy_matrix("Y", Y.transpose());
      }
      if (options.check_gradients)
      {
        DY = loss->gradient(Y, T);
        auto f = [loss, &Y, &T]() { return loss->value(Y, T); };
        check_gradient("DY", f, Y, DY, options.check_gradients_step);
      }
      else
      {
        DY = loss->gradient(Y, T) / options.batch_size;  // pytorch does it like this
      }
      M.backpropagate(Y, DY);
      if (options.check_gradients)
      {
        M.check_gradients(loss, T, options.check_gradients_step);
      }
      M.optimize(eta);
    }
    double seconds = watch.seconds();
    total_training_time += seconds;
    compute_statistics_batch(M, loss, data, options.batch_size, epoch, options.statistics, seconds);
  }
  double test_accuracy = compute_accuracy_batch(M, data.Xtest, data.Ttest, options.batch_size);
  std::cout << fmt::format("Accuracy of the network on the {} test examples: {:4.2f}%", data.Xtest.cols(), test_accuracy * 100.0) << std::endl;
  std::cout << fmt::format("Total training time for the {} epochs: {:7.4f}s\n", options.epochs, total_training_time);
  return {test_accuracy, total_training_time};
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_TRAINING_H
