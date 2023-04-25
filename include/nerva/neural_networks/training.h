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
#include "nerva/neural_networks/regrow.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/print.h"
#include "nerva/utilities/timer.h"
#include "nerva/neural_networks/global_timer.h"
#include "fmt/format.h"
#include <algorithm>
#include <iomanip>

namespace nerva {

template <typename EigenMatrix>
double compute_accuracy(multilayer_perceptron& M, const EigenMatrix& Xtest, const EigenMatrix& Ttest, long Q)
{
  global_timer_suspend();

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
    auto Xbatch = Xtest(Eigen::indexing::all, batch);
    auto Tbatch = Ttest(Eigen::indexing::all, batch);
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

  global_timer_resume();

  return static_cast<double>(total_correct) / N;
}

inline
double compute_loss(multilayer_perceptron& M, const std::shared_ptr<loss_function>& loss, const eigen::matrix& X, const eigen::matrix& T, long Q)
{
  global_timer_suspend();

  if (has_nan(M))
  {
    print_model_info(M);
    throw std::runtime_error("the multilayer perceptron contains NaN values");
  }

  long N = X.cols(); // the number of examples
  long L = T.rows(); // the number of outputs
  auto K = N / Q;    // the number of batches
  double total_loss = 0.0;
  eigen::matrix Ybatch(L, Q);

  for (long k = 0; k < K; k++)
  {
    auto batch = Eigen::seqN(k * Q, Q);
    auto Xbatch = X(Eigen::indexing::all, batch);
    auto Tbatch = T(Eigen::indexing::all, batch);
    M.feedforward(Xbatch, Ybatch);
    total_loss += loss->value(Ybatch, Tbatch);
  }

  global_timer_resume();

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

template <typename DataSet = datasets::dataset>
class stochastic_gradient_descent_algorithm
{
  protected:
    multilayer_perceptron& M;
    DataSet& data;
    const sgd_options& options;
    const std::shared_ptr<loss_function>& loss;
    const std::shared_ptr<learning_rate_scheduler>& learning_rate;
    std::mt19937& rng;
    utilities::map_timer timer;

  public:
    stochastic_gradient_descent_algorithm(multilayer_perceptron& M_,
                                          DataSet& data_,
                                          const sgd_options& options_,
                                          const std::shared_ptr<loss_function>& loss_,
                                          const std::shared_ptr<learning_rate_scheduler>& learning_rate_,
                                          std::mt19937& rng_
                                         )
      : M(M_),
        data(data_),
        options(options_),
        loss(loss_),
        learning_rate(learning_rate_),
        rng(rng_)
    {}

    /// \brief Event function that is called at the start of training
    virtual void on_start_training()
    {}

    /// \brief Event function that is called at the end of training
    virtual void on_end_training()
    {}

    /// \brief Event function that is called at the start of each epoch
    virtual void on_start_epoch(unsigned int epoch)
    {}

    /// \brief Event function that is called at the end of each epoch
    virtual void on_end_epoch(unsigned int epoch)
    {}

    /// \brief Event function that is called at the start of each batch
    virtual void on_start_batch()
    {}

    /// \brief Event function that is called at the end of each batch
    virtual void on_end_batch()
    {}

    /// \brief Returns the sum of the measured times for the epochs
    [[nodiscard]] double compute_training_time() const
    {
      double result = 0;
      for (const auto& [key, value]: timer.values())
      {
        if (utilities::starts_with(key, "epoch"))
        {
          result += timer.seconds(key);
        }
      }
      return result;
    }

    std::pair<double, double> run()
    {
      on_start_training();

      long N = data.Xtrain.cols(); // the number of examples
      long L = data.Ttrain.rows(); // the number of outputs
      std::vector<long> I(N);
      std::iota(I.begin(), I.end(), 0);
      eigen::matrix Y(L, options.batch_size);
      long K = N / options.batch_size; // the number of batches
      scalar eta = learning_rate->operator()(0);

      compute_statistics(M, eta, loss, data, options.batch_size, -1, options.statistics, 0.0);

      for (unsigned int epoch = 0; epoch < options.epochs; ++epoch)
      {
        on_start_epoch(epoch);
        std::string epoch_label = fmt::format("epoch{}", epoch);
        timer.start(epoch_label);

        if (options.shuffle)
        {
          std::shuffle(I.begin(), I.end(), rng);      // shuffle the examples at the start of each epoch
        }

        eta = learning_rate->operator()(epoch);       // update the learning at the start of each epoch

        eigen::matrix DY(L, options.batch_size);

        for (long k = 0; k < K; k++)
        {
          on_start_batch();

          eigen::eigen_slice batch(I.begin() + k * options.batch_size, options.batch_size);
          auto X = data.Xtrain(Eigen::indexing::all, batch);
          auto T = data.Ttrain(Eigen::indexing::all, batch);
          M.feedforward(X, Y);

          if (options.gradient_step > 0)
          {
            DY = loss->gradient(Y, T);
            auto f = [this, &Y, &T]() { return loss->value(Y, T); };
            check_gradient("DY", f, Y, DY, options.gradient_step);
          }
          else
          {
            DY = loss->gradient(Y, T) / options.batch_size;  // pytorch does it like this
          }

          if (options.debug)
          {
            std::cout << "epoch: " << epoch << " batch: " << k << std::endl;
            print_model_info(M);
            eigen::print_numpy_matrix("X", X.transpose());
            eigen::print_numpy_matrix("Y", Y.transpose());
            eigen::print_numpy_matrix("DY", DY.transpose());
          }

//          if (eigen::has_nan(Y))
//          {
//            eigen::print_numpy_matrix("Y", Y.transpose());
//            throw std::runtime_error("the output Y contains NaN values");
//          }
//
//          if (eigen::has_nan(DY))
//          {
//            eigen::print_numpy_matrix("DY", DY.transpose());
//            throw std::runtime_error("the gradient DY contains NaN values");
//          }

          M.backpropagate(Y, DY);

          if (options.gradient_step > 0)
          {
            M.check_gradients(loss, T, options.gradient_step);
          }

          M.optimize(eta);

          on_end_batch();
        }

        timer.stop(epoch_label);
        double seconds = timer.seconds(epoch_label);
        compute_statistics(M, eta, loss, data, options.batch_size, epoch, options.statistics, seconds);

        on_end_epoch(epoch);
      }

      double test_accuracy = compute_accuracy(M, data.Xtest, data.Ttest, options.batch_size);
      double training_time = compute_training_time();
      // std::cout << fmt::format("Accuracy of the network on the {} test examples: {:.2f}%", data.Xtest.cols(), test_accuracy * 100.0) << std::endl;
      std::cout << fmt::format("Total training time for the {} epochs: {:.8f}s\n", options.epochs, training_time);

      on_end_training();

      return {test_accuracy, training_time};
    }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_TRAINING_H
