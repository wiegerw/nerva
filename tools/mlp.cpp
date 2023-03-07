// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/examples/multilayer_perceptron.cpp
/// \brief add your file description here.

#include "nerva/neural_networks/multilayer_perceptron.h"
#include "omp.h"
#include "nerva/datasets/dataset.h"
#include "nerva/datasets/make_dataset.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/sgd_options.h"
#include "nerva/neural_networks/parse_layer.h"
#include "nerva/neural_networks/training.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/string_utility.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>

#ifdef NERVA_ENABLE_PROFILING
#include <valgrind/callgrind.h>
#endif

using namespace nerva;

// Does an attempt to print the original command line call.
// N.B. It does not handle nested quotes.
inline
void print_command_line_call(int argc, const char* argv[])
{
  for (int i = 0; i < argc; i++) {
    if (std::string(argv[i]).find_first_of(" ()") != std::string::npos)
    {
      std::cout << "\"" << argv[i] << "\"";
    }
    else
    {
      std::cout << argv[i];
    }
    if (i < argc - 1)
    {
      std::cout << " ";
    }
  }
  std::cout << "\n\n";
}

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

template <typename RandomNumberGenerator>
void set_weights(multilayer_perceptron& M, std::string weights_initialization, const std::string& architecture, RandomNumberGenerator rng)
{
  // use default weights if the weights initialization was unspecified
  if (weights_initialization.empty())
  {
    auto B_count = std::count(architecture.begin(), architecture.end(), 'B');
    std::size_t count = M.layers.size() - B_count;
    weights_initialization = std::string(count, 'd');
  }

  unsigned int index = 0;
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      initialize_weights(parse_weight_char(weights_initialization[index++]), dlayer->W, dlayer->b, rng);
    }
    else if (auto mlayer = dynamic_cast<linear_layer<mkl::sparse_matrix_csr<scalar>>*>(layer.get()))
    {
      initialize_weights(parse_weight_char(weights_initialization[index++]), mlayer->W, mlayer->b, rng);
    }
 }
}

inline
std::vector<double> parse_comma_separated_real_numbers(const std::string& text)
{
  std::vector<double> result;
  for (const std::string& word: utilities::regex_split(text, ","))
  {
    result.push_back(static_cast<double>(parse_double(word)));
  }
  return result;
}

inline
std::vector<double> compute_densities(const std::string& architecture, const std::vector<std::size_t>& sizes, double density)
{
  if (architecture.size() + 1 != sizes.size())
  {
    throw std::runtime_error("Unexpected number of sizes in compute_densities");
  }

  std::vector<double> densities;

  if (density == 1)
  {
    for (char a: architecture)
    {
      if (is_linear_layer(a))
      {
        densities.push_back(density);
      }
    }
  }
  else
  {
    densities = compute_sparse_layer_densities(density, sizes);
  }
  return densities;
}

// D is the input size of the neural network
// K is the output size of the neural network
inline
std::vector<std::size_t> compute_sizes(const std::vector<std::size_t>& layer_sizes, const std::string& architecture)
{
  std::size_t D = layer_sizes.front(); // the number of features
  std::size_t K = layer_sizes.back(); // the number of outputs
  auto B_count = std::count(architecture.begin(), architecture.end(), 'B');

  if (layer_sizes.size() != architecture.size() - B_count + 1)
  {
    throw std::runtime_error("The number of layer sizes does not match with the given architecture.");
  }

  std::vector<std::size_t> sizes = { D };
  int index = 1;
  for (char ch: architecture.substr(0, architecture.size() - 1))
  {
    if (ch == 'B')
    {
      sizes.push_back(sizes.back());
    }
    else
    {
      sizes.push_back(layer_sizes[index++]);
    }
  }
  sizes.push_back(K);

  return sizes;
}

inline
void check_options(const mlp_options& options, const std::vector<std::size_t>& layer_sizes)
{
  if (options.architecture.empty())
  {
    throw std::runtime_error("There should be at least one layer.");
  }
  
  if (options.architecture.back() == 'B')
  {
    throw std::runtime_error("The last layer should not be a batch normalization layer.");
  }

  auto B_count = std::count(options.architecture.begin(), options.architecture.end(), 'B');
  if (layer_sizes.size() != options.architecture.size() - B_count + 1)
  {
    throw std::runtime_error("The number of hidden layer sizes does not match with the given architecture.");
  }

  if (!options.weights_initialization.empty())
  {
    if (options.weights_initialization.size() != options.architecture.size() - B_count)
    {
      throw std::runtime_error("The weight initialization does not match with the given architecture.");
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
    std::string layer_sizes_text;
    std::string densities_text;
    std::string preprocessed_dir;  // a directory containing a dataset for every epoch
    bool no_shuffle = false;
    bool no_statistics = false;
    bool info = false;

    void add_options(lyra::cli& cli) override
    {
      // randomness
      cli |= lyra::opt(options.seed, "value")["--seed"]("A seed value for the random generator.");

      // model parameters
      cli |= lyra::opt(layer_sizes_text, "value")["--sizes"]("A comma separated list of layer sizes");
      cli |= lyra::opt(densities_text, "value")["--densities"]("A comma separated list of sparse layer densities");
      cli |= lyra::opt(options.density, "value")["--overall-density"]("The overall density level of the sparse layers");
      cli |= lyra::opt(options.architecture, "value")["--architecture"]("The architecture of the multilayer perceptron e.g. RRL,\nwhere R=ReLU, S=Sigmoid, L=Linear, B=Batchnorm, Z=Softmax");
      cli |= lyra::opt(options.dropout, "value")["--dropout"]("The dropout rate for the the linear layers (use 0 for no dropout)");

      // training
      cli |= lyra::opt(options.algorithm, "value")["--algorithm"]("The algorithm (sgd)");
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

      // regrow (experimental!)
      cli |= lyra::opt(options.regrow_rate, "value")["--zeta"]("The regrow rate, use 0 for no regrow.");
      cli |= lyra::opt(options.regrow_separate_positive_negative)["--separate"]("Separate negative and positive weights for regrow");

      cli |= lyra::opt(options.threads, "value")["--threads"]("The number of threads used by Eigen.");
      cli |= lyra::opt(options.gradient_step, "value")["--gradient-step"]("If positive, gradient checks will be done with the given step size");
    }

    std::string description() const override
    {
      return "Multilayer perceptron test";
    }

    bool run() override
    {
      options.debug = is_debug();
      if (no_shuffle)
      {
        options.shuffle = false;
      }
      if (no_statistics)
      {
        options.statistics = false;
      }
      std::vector<std::size_t> layer_sizes = parse_comma_separated_numbers(layer_sizes_text);
      std::vector<double> densities = parse_comma_separated_real_numbers(densities_text);
      check_options(options, layer_sizes);

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

      options.sizes = compute_sizes(layer_sizes, options.architecture);

      if (densities.empty())
      {
        densities = compute_densities(options.architecture, options.sizes, options.density);
      }
      options.densities = densities;
      options.info();

      if (options.threads >= 1 && options.threads <= 8)
      {
        omp_set_num_threads(options.threads);
      }

      // construct the multilayer perceptron M
      multilayer_perceptron M;
      unsigned int densities_index = 0;
      for (std::size_t i = 0; i < options.architecture.size(); i++)
      {
        char a = options.architecture[i];
        if (!is_linear_layer(a) || densities[densities_index] == 1.0)
        {
          M.layers.push_back(parse_dense_layer(a, options.sizes[i], options.sizes[i+1], options, rng));
        }
        else
        {
          double density = densities[densities_index++];
          M.layers.push_back(parse_sparse_layer(a, options.sizes[i], options.sizes[i+1], density, options, rng));
        }

        if (auto layer = dynamic_cast<dense_linear_layer*>(M.layers.back().get()))
        {
          set_optimizer(*layer, options.optimizer);
        }
        else if (auto mlayer = dynamic_cast<linear_layer<mkl::sparse_matrix_csr<scalar>>*>(M.layers.back().get()))
        {
          set_optimizer(*mlayer, options.optimizer);
        }
      }

      std::shared_ptr<loss_function> loss = parse_loss_function(options.loss_function);
      std::shared_ptr<learning_rate_scheduler> learning_rate = parse_learning_rate_scheduler(options.learning_rate_scheduler);

      if (load_weights_file.empty())
      {
        set_weights(M, options.weights_initialization, options.architecture, rng);
      }
      else
      {
        load_weights(M, load_weights_file);
      }

      if (!save_weights_file.empty())
      {
        save_weights(M, save_weights_file);
      }

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

      if (options.algorithm == "sgd")
      {
        if (!preprocessed_dir.empty())
        {
          stochastic_gradient_descent_preprocessed(M, loss, preprocessed_dir, options, learning_rate, rng);
        }
        else
        {
          stochastic_gradient_descent(M, loss, dataset, options, learning_rate, rng);
        }
      }
      else
      {
        throw std::runtime_error("Unknown algorithm '" + options.algorithm + "'");
      }

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
  print_command_line_call(argc, argv);
  return tool().execute(argc, argv);
}
