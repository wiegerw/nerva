// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/tools/mlp.cpp
/// \brief add your file description here.

#include "nerva/datasets/dataset.h"
#include "nerva/datasets/make_dataset.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/loss_functions_colwise.h"
#include "nerva/neural_networks/mlp_algorithms.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/parse_layer.h"
#include "nerva/neural_networks/regrow.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/neural_networks/training.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/string_utility.h"
#include "omp.h"
#include <iostream>
#include <random>

#ifdef NERVA_ENABLE_PROFILING
#include <valgrind/callgrind.h>
#endif

using namespace nerva;

inline
auto parse_linear_layer_densities(const std::string& densities_text,
                                  double overall_density,
                                  const std::vector<std::size_t>& linear_layer_sizes) -> std::vector<double>
{
  std::vector<double> densities = parse_comma_separated_real_numbers(densities_text);
  auto n = linear_layer_sizes.size() - 1;  // the number of linear layers

  if (densities.empty())
  {
    if (overall_density == 1)
    {
      densities = std::vector<double>(n, 1);
    }
    else
    {
      densities = compute_sparse_layer_densities(overall_density, linear_layer_sizes);
    }
  }

  if (densities.size() != n)
  {
    throw std::runtime_error("the number of densities does not match with the number of linear layers");
  }

  return densities;
}

inline
auto parse_linear_layer_dropouts(const std::string& dropouts_text, std::size_t linear_layer_count) -> std::vector<double>
{
  std::vector<double> dropouts = parse_comma_separated_real_numbers(dropouts_text);

  if (dropouts.empty())
  {
    return std::vector<double>(linear_layer_count, 0.0);
  }

  if (dropouts.size() != linear_layer_count)
  {
    throw std::runtime_error("the number of dropouts does not match with the number of linear layers");
  }

  return dropouts;
}

inline
auto parse_init_weights(const std::string& text, std::size_t linear_layer_count) -> std::vector<std::string>
{
  auto n = linear_layer_count;

  std::vector<std::string> words = utilities::regex_split(utilities::trim_copy(text), ";");
  if (words.size() == 1)
  {
    auto init = words.front();
    return { n, init };
  }

  if (words.size() != n)
  {
    throw std::runtime_error(fmt::format("the number of weight initializers ({}) does not match with the number of linear layers ({})", words.size(), n));
  }

  return words;
}

inline
auto parse_optimizers(const std::string& text, std::size_t count) -> std::vector<std::string>
{
  std::vector<std::string> words = utilities::regex_split(text, "\\s*,\\s*");
  if (words.empty())
  {
    return {count, "GradientDescent"};
  }
  if (words.size() == 1)
  {
    return {count, words.front()};
  }
  if (words.size() != count)
  {
    throw std::runtime_error(fmt::format("expected {} optimizers instead of {}", count, words.size()));
  }
  return words;
}

inline
void set_optimizers(multilayer_perceptron& M, const std::string& optimizer)
{
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      set_linear_layer_optimizer(*dlayer, optimizer);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      set_linear_layer_optimizer(*slayer, optimizer);
    }
  }
}

class sgd_algorithm: public stochastic_gradient_descent_algorithm<datasets::dataset>
{
  protected:
    std::filesystem::path preprocessed_dir;
    std::shared_ptr<prune_and_grow> regrow;

    using super = stochastic_gradient_descent_algorithm<datasets::dataset>;
    using super::data;
    using super::M;
    using super::rng;
    using super::timer;

  public:
    sgd_algorithm(multilayer_perceptron& M,
                  datasets::dataset& data,
                  const sgd_options& options,
                  const std::shared_ptr<loss_function>& loss,
                  const std::shared_ptr<learning_rate_scheduler>& learning_rate,
                  std::mt19937& rng,
                  const std::string& preprocessed_dir_,
                  const std::shared_ptr<prune_function>& prune,
                  const std::shared_ptr<grow_function>& grow
    )
      : super(M, data, options, loss, learning_rate, rng),
        preprocessed_dir(preprocessed_dir_)
    {
      if (prune)
      {
        regrow = std::make_shared<prune_and_grow>(prune, grow);
      }
    }

    /// \brief Reloads the dataset if a directory with preprocessed data was specified.
    void reload_data(unsigned int epoch)
    {
      if (!preprocessed_dir.empty())
      {
        data.load((preprocessed_dir / ("epoch" + std::to_string(epoch) + ".npz")).native());
      }
    }

    void on_start_training() override
    {
      reload_data(0);
    }

    void on_start_epoch(unsigned int epoch) override
    {
      if (epoch > 0)
      {
        reload_data(epoch);
      }

      renew_dropout_masks(M, rng);

      if (epoch > 0 && regrow)
      {
        (*regrow)(M);
      }
    }

    void on_end_epoch(unsigned int epoch) override
    {
      // print_srelu_layers(M);
    }
};

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
    std::string dropouts_text;
    std::string layer_specifications_text;
    std::string init_weights_text = "None";
    double overall_density = 1;
    std::string preprocessed_dir;  // a directory containing a dataset for every epoch
    bool no_shuffle = false;
    bool no_statistics = false;
    bool info = false;
    bool use_global_timer = false;

    // pruning + growing
    std::string prune_strategy;
    std::string grow_strategy = "Random";
    std::string grow_weights = "Zero";

    void add_options(lyra::cli& cli) override
    {
      // randomness
      cli |= lyra::opt(options.seed, "value")["--seed"]("A seed value for the random generator.");

      // model parameters
      cli |= lyra::opt(linear_layer_sizes_text, "value")["--sizes"]("A comma separated list of layer sizes");
      cli |= lyra::opt(densities_text, "value")["--densities"]("A comma separated list of sparse layer densities");
      cli |= lyra::opt(dropouts_text, "value")["--dropouts"]("A comma separated list of dropout rates");
      cli |= lyra::opt(overall_density, "value")["--overall-density"]("The overall density level of the sparse layers");
      cli |= lyra::opt(layer_specifications_text, "value")["--layers"]("A semi-colon separated lists of layers. The following layers are supported: "
                                                                  "Linear, ReLU, Sigmoid, Softmax, LogSoftmax, HyperbolicTangent, BatchNorm, "
                                                                  "AllRelu(<alpha>), TReLU(<epsilon>)");

      // training
      cli |= lyra::opt(options.epochs, "value")["--epochs"]("The number of epochs (default: 100)");
      cli |= lyra::opt(options.batch_size, "value")["--batch-size"]("The batch size of the training algorithm");
      cli |= lyra::opt(no_shuffle)["--no-shuffle"]("Do not shuffle the dataset during training.");
      cli |= lyra::opt(no_statistics)["--no-statistics"]("Do not compute statistics during training.");

      // optimizer
      cli |= lyra::opt(options.optimizer, "value")["--optimizers"]("The optimizer (GradientDescent, Momentum(<mu>), Nesterov(<mu>))");

      // learning rate
      cli |= lyra::opt(options.learning_rate_scheduler, "value")["--learning-rate"]("The learning rate scheduler (default: constant(0.0001))");

      // loss function
      cli |= lyra::opt(options.loss_function, "value")["--loss"]("The loss function (squared-error, cross-entropy, logistic-cross-entropy)");

      // weights
      cli |= lyra::opt(init_weights_text, "value")["--init-weights"]("The weight initialization (default, he, uniform, xavier, normalized_xavier, uniform)");
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
      cli |= lyra::opt(grow_strategy, "strategy")["--grow"]("The growing strategy: (default: Random)");
      cli |= lyra::opt(grow_weights, "value")["--grow-weights"]("The weight function used for growing x=Xavier, X=XavierNormalized, ...");

      // miscellaneous
      cli |= lyra::opt(options.threads, "value")["--threads"]("The number of threads used by Eigen.");
      cli |= lyra::opt(options.gradient_step, "value")["--gradient-step"]("If positive, gradient checks will be done with the given step size");
    }

    auto description() const -> std::string override
    {
      return "Multilayer perceptron test";
    }

    auto run() -> bool override
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
      if (options.threads >= 1 && options.threads <= 8)
      {
        omp_set_num_threads(options.threads);
      }

      std::mt19937 rng{options.seed};

      datasets::dataset dataset;
      if (!options.dataset.empty())
      {
        NERVA_LOG(log::verbose) << "Loading dataset " << options.dataset << '\n';
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
      auto linear_layer_sizes = parse_comma_separated_numbers(linear_layer_sizes_text);
      auto linear_layer_count = linear_layer_sizes.size() - 1;
      auto linear_layer_densities = parse_linear_layer_densities(densities_text, overall_density, linear_layer_sizes);
      auto linear_layer_dropouts = parse_linear_layer_dropouts(dropouts_text, linear_layer_count);
      auto linear_layer_weights = parse_init_weights(init_weights_text, linear_layer_count);
      auto optimizers = parse_optimizers(options.optimizer, layer_specifications.size());

      // construct the multilayer perceptron M
      multilayer_perceptron M;
      M.layers = make_layers(layer_specifications, linear_layer_sizes, linear_layer_densities, linear_layer_dropouts, linear_layer_weights, optimizers, options.batch_size, rng);

      if (!load_weights_file.empty())
      {
        load_weights_and_bias(M, load_weights_file);
      }

      if (!save_weights_file.empty())
      {
        save_weights_and_bias(M, save_weights_file);
      }

      std::shared_ptr<loss_function> loss = parse_loss_function(options.loss_function);
      std::shared_ptr<learning_rate_scheduler> learning_rate = parse_learning_rate_scheduler(options.learning_rate_scheduler);
      std::shared_ptr<prune_function> prune = parse_prune_function(prune_strategy);
      std::shared_ptr<grow_function> grow = parse_grow_function(grow_strategy, parse_weight_initialization(grow_weights), rng);

      if (info)
      {
        dataset.info();
        M.info("before training");
      }

      std::cout << "=== Nerva c++ model ===" << "\n";
      std::cout << M.to_string();
      std::cout << "loss = " << loss->to_string() << "\n";
      std::cout << "scheduler = " << learning_rate->to_string() << "\n";
      std::cout << "layer densities: " << layer_density_info(M) << "\n\n";

      sgd_algorithm algorithm(M, dataset, options, loss, learning_rate, rng, preprocessed_dir, prune, grow);

#ifdef NERVA_ENABLE_PROFILING
      CALLGRIND_START_INSTRUMENTATION;
#endif

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

  public:
    virtual ~tool() = default;
};

auto main(int argc, const char* argv[]) -> int
{
  pybind11::scoped_interpreter guard{};
  return tool().execute(argc, argv);
}
