// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/tools/mlp.cpp
/// \brief add your file description here.

#include "nerva/neural_networks/multilayer_perceptron.h"
#include "omp.h"
#include "nerva/datasets/dataset.h"
#include "nerva/datasets/make_dataset.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/parse_layer.h"
#include "nerva/neural_networks/regrow.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/neural_networks/training.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/string_utility.h"
#include <iostream>
#include <random>

#ifdef NERVA_ENABLE_PROFILING
#include <valgrind/callgrind.h>
#endif

using namespace nerva;

inline
weight_initialization parse_weight_char(char c)
{
  if (c == 'd')
  {
    return weight_initialization::default_;
  }
  else if (c == 'h')
  {
    return weight_initialization::he;
  }
  else if (c == 'x')
  {
    return weight_initialization::xavier;
  }
  else if (c == 'X')
  {
    return weight_initialization::xavier_normalized;
  }
  else if (c == 'u')
  {
    return weight_initialization::uniform;
  }
  else if (c == 'p')
  {
    return weight_initialization::pytorch;
  }
  else if (c == 't')
  {
    return weight_initialization::tensorflow;
  }
  throw std::runtime_error(std::string("could not parse weight char '") + c + "'");
}

inline
std::vector<weight_initialization> parse_weights(std::string weights_initialization, const std::vector<std::string>& linear_layer_specifications)
{
  // use default weights if the weights initialization was unspecified
  if (weights_initialization.empty())
  {
    weights_initialization = std::string(linear_layer_specifications.size(), 'd');
  }

  std::vector<weight_initialization> weights;
  for (char c: weights_initialization)
  {
    weights.push_back(parse_weight_char(c));
  }

  return weights;
}

inline
void set_optimizers(multilayer_perceptron& M, const std::string& optimizer)
{
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      set_optimizer(*dlayer, optimizer);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      set_optimizer(*slayer, optimizer);
    }
  }
}

class tool: public command_line_tool
{
  protected:
    mlp_options options;
    std::string load_weights_file;
    std::string save_weights_file;
    std::string load_dataset_file;
    std::string save_dataset_file;
    std::string linear_layer_sizes_text;
    std::string densities_text;
    std::string layer_specifications_text;
    double overall_density = 1;
    std::string preprocessed_dir;  // a directory containing a dataset for every epoch
    bool no_shuffle = false;
    bool no_statistics = false;
    bool info = false;
    bool use_global_timer = false;

    // pruning + growing
    std::string prune_strategy;
    std::string grow_strategy = "Random";
    unsigned int regrow_interval = 0;

    void add_options(lyra::cli& cli) override
    {
      // randomness
      cli |= lyra::opt(options.seed, "value")["--seed"]("A seed value for the random generator.");

      // model parameters
      cli |= lyra::opt(linear_layer_sizes_text, "value")["--sizes"]("A comma separated list of layer sizes");
      cli |= lyra::opt(densities_text, "value")["--densities"]("A comma separated list of sparse layer densities");
      cli |= lyra::opt(overall_density, "value")["--overall-density"]("The overall density level of the sparse layers");
      cli |= lyra::opt(layer_specifications_text, "value")["--layers"]("A semi-colon separated lists of layers. The following layers are supported: "
                                                                  "Linear, ReLU, Sigmoid, Softmax, LogSoftmax, HyperbolicTangent, BatchNorm, "
                                                                  "Dropout(<rate>), AllRelu(<alpha>), TReLU(<epsilon>)");

      // training
      cli |= lyra::opt(options.epochs, "value")["--epochs"]("The number of epochs (default: 100)");
      cli |= lyra::opt(options.batch_size, "value")["--batch-size"]("The batch size of the training algorithm");
      cli |= lyra::opt(no_shuffle)["--no-shuffle"]("Do not shuffle the dataset during training.");
      cli |= lyra::opt(no_statistics)["--no-statistics"]("Do not compute statistics during training.");

      // optimizer
      cli |= lyra::opt(options.optimizer, "value")["--optimizer"]("The optimizer (gradient-descent, momentum(<mu>), nesterov(<mu>))");

      // learning rate
      cli |= lyra::opt(options.learning_rate_scheduler, "value")["--learning-rate"]("The learning rate scheduler (default: constant(0.0001))");

      // loss function
      cli |= lyra::opt(options.loss_function, "value")["--loss"]("The loss function (squared-error, cross-entropy, logistic-cross-entropy)");

      // weights
      cli |= lyra::opt(options.weights_initialization, "value")["--weights"]("The weight initialization (default, he, uniform, xavier, normalized_xavier, uniform)");
      cli |= lyra::opt(load_weights_file, "value")["--load-weights"]("Loads the weights and bias from a file in .npz format");
      cli |= lyra::opt(save_weights_file, "value")["--save-weights"]("Saves the weights and bias to a file in .npz format");

      // dataset
      cli |= lyra::opt(options.dataset, "value")["--dataset"]("The dataset (chessboard, spirals, square, sincos)");
      cli |= lyra::opt(load_dataset_file, "value")["--load-dataset"]("Loads the dataset from a file in .npz format");
      cli |= lyra::opt(save_dataset_file, "value")["--save-dataset"]("Saves the dataset to a file in .npz format");
      cli |= lyra::opt(options.dataset_size, "value")["--size"]("The size of the dataset (default: 1000)");
      cli |= lyra::opt(options.normalize_data)["--normalize"]("Normalize the data");
      cli |= lyra::opt(preprocessed_dir, "value")["--preprocessed"]("A directory containing the files epoch<nnn>.npz");

      // print options
      cli |= lyra::opt(options.precision, "value")["--precision"]("The precision that is used for printing.");
      cli |= lyra::opt(info)["--info"]("print some info about the multilayer_perceptron's");
      cli |= lyra::opt(use_global_timer)["--timer"]("print timer messages");

      // pruning + growing
      cli |= lyra::opt(prune_strategy, "strategy")["--prune"]("The pruning strategy: Magnitude(<drop_fraction>), SET(<drop_fraction>) or Threshold(<value>)");
      cli |= lyra::opt(regrow_interval, "value")["--prune-interval"]("The number of batches between pruning + growing weights (default: 1 epoch)");
      cli |= lyra::opt(grow_strategy, "strategy")["--grow"]("The growing strategy: (default: Random)");

      // miscellaneous
      cli |= lyra::opt(options.threads, "value")["--threads"]("The number of threads used by Eigen.");
      cli |= lyra::opt(options.gradient_step, "value")["--gradient-step"]("If positive, gradient checks will be done with the given step size");
    }

    std::string description() const override
    {
      return "Multilayer perceptron test";
    }

    bool run() override
    {
      NERVA_LOG(log::verbose) << command_line_call() << "\n\n";

      options.debug = is_debug();
      if (no_shuffle)
      {
        options.shuffle = false;
      }
      if (no_statistics)
      {
        options.statistics = false;
      }
      if (use_global_timer)
      {
        global_timer_enable();
      }

      std::mt19937 rng{options.seed};

      datasets::dataset dataset;
      if (!options.dataset.empty())
      {
        NERVA_LOG(log::verbose) << "Loading dataset " << options.dataset << std::endl;
        dataset = datasets::make_dataset(options.dataset, options.dataset_size, rng);
        if (!save_dataset_file.empty())
        {
          dataset.save(save_dataset_file);
        }
      }
      else if (!load_dataset_file.empty())
      {
        dataset.load(load_dataset_file);
      }

      auto layer_specifications = parse_layers(layer_specifications_text);
      auto linear_layer_specifications = filter_sequence(layer_specifications, is_linear_layer);
      auto linear_layer_sizes = compute_linear_layer_sizes(linear_layer_sizes_text, linear_layer_specifications);
      auto linear_layer_densities = compute_linear_layer_densities(densities_text, overall_density, linear_layer_specifications, linear_layer_sizes);

      if (options.threads >= 1 && options.threads <= 8)
      {
        omp_set_num_threads(options.threads);
      }

      // construct the multilayer perceptron M
      multilayer_perceptron M;
      M.layers = construct_layers(layer_specifications, linear_layer_sizes, linear_layer_densities, options.batch_size, rng);
      set_optimizers(M, options.optimizer);

      if (load_weights_file.empty())
      {
        set_support_random(M, linear_layer_densities, rng);
        auto weights = parse_weights(options.weights_initialization, linear_layer_specifications);
        set_weights_and_bias(M, weights, rng);
      }
      else
      {
        load_weights_and_bias(M, load_weights_file);
      }

      if (!save_weights_file.empty())
      {
        save_weights_and_bias(M, save_weights_file);
      }

      std::shared_ptr<loss_function> loss = parse_loss_function(options.loss_function);
      std::shared_ptr<learning_rate_scheduler> learning_rate = parse_learning_rate_scheduler(options.learning_rate_scheduler);

      //std::shared_ptr<prune_function> prune = parse_prune_function(prune_strategy);
      //std::shared_ptr<grow_function> grow = parse_grow_function(grow_strategy, weight_initialization::xavier_normalized, rng);
      //std::shared_ptr<regrow_function> regrow = std::make_shared<prune_and_grow>(prune, grow);

      std::shared_ptr<regrow_function> regrow = parse_regrow_function(prune_strategy, weight_initialization::xavier_normalized, rng);

      if (info)
      {
        dataset.info();
        M.info("before training");
      }

#ifdef NERVA_ENABLE_PROFILING
      CALLGRIND_START_INSTRUMENTATION;
#endif

      std::cout << "=== Nerva c++ model ===" << "\n";
      std::cout << M.to_string();
      std::cout << "loss = " << loss->to_string() << "\n";
      std::cout << "scheduler = " << learning_rate->to_string() << "\n";
      std::cout << "layer densities: " << layer_density_info(M) << "\n\n";

      stochastic_gradient_descent_algorithm algorithm(M, dataset, options, loss, learning_rate, rng, preprocessed_dir, regrow);
      algorithm.run();

#ifdef NERVA_ENABLE_PROFILING
      CALLGRIND_STOP_INSTRUMENTATION;
      CALLGRIND_DUMP_STATS;
#endif

      if (info)
      {
        M.info("after training");
      }

      return true;
    }
};

int main(int argc, const char* argv[])
{
  pybind11::scoped_interpreter guard{};
  return tool().execute(argc, argv);
}
