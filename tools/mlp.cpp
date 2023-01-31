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
  throw std::runtime_error(std::string("Error: could not parse weight char '") + c + "'");
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
std::vector<scalar> parse_comma_separated_real_numbers(const std::string& text)
{
  std::vector<scalar> result;
  for (const std::string& word: utilities::regex_split(text, ","))
  {
    result.push_back(static_cast<scalar>(parse_double(word)));
  }
  return result;
}

inline
std::vector<scalar> compute_densities(const std::string& architecture, const std::vector<std::size_t>& sizes, scalar density)
{
  if (architecture.size() + 1 != sizes.size())
  {
    throw std::runtime_error("Unexpected number of sizes in compute_densities");
  }

  std::vector<scalar> densities;

  if (density == scalar(1))
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
    std::vector<std::pair<long, long>> sparse_linear_layer_shapes;
    for (std::size_t i = 0; i < architecture.size(); i++)
    {
      char a = architecture[i];
      if (is_linear_layer(a))
      {
        sparse_linear_layer_shapes.emplace_back(sizes[i], sizes[i+1]);
      }
    }
    densities = compute_sparse_layer_densities(density, sparse_linear_layer_shapes);
  }
  return densities;
}

// D is the input size of the neural network
// K is the output size of the neural network
inline
std::vector<std::size_t> compute_sizes(std::size_t D, std::size_t K, const std::vector<std::size_t>& hidden_layer_sizes, const std::string& architecture)
{
  auto B_count = std::count(architecture.begin(), architecture.end(), 'B');
  if (hidden_layer_sizes.size() != architecture.size() - B_count - 1)
  {
    throw std::runtime_error("The number of hidden layer sizes does not match with the given architecture.");
  }

  std::vector<std::size_t> sizes = { D };
  int index = 0;
  for (char ch: architecture.substr(0, architecture.size() - 1))
  {
    if (ch == 'B')
    {
      sizes.push_back(sizes.back());
    }
    else
    {
      sizes.push_back(hidden_layer_sizes[index++]);
    }
  }
  sizes.push_back(K);

  return sizes;
}

inline
void check_options(const mlp_options& options, const std::vector<std::size_t>& hidden_layer_sizes)
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
  if (hidden_layer_sizes.size() != options.architecture.size() - B_count - 1)
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
    std::string datadir;
    std::string import_weights_file;
    std::string export_weights_file;
    std::string hidden_layer_sizes_text;
    std::string densities_text;
    bool no_shuffle = false;
    bool no_statistics = false;
    bool info = false;

    void add_options(lyra::cli& cli) override
    {
      cli |= lyra::opt(options.algorithm, "value")["--algorithm"]("The algorithm (sgd, minibatch, minibatch_sgd)");
      cli |= lyra::opt(options.dataset, "value")["--dataset"]("The dataset (chessboard, spirals, square, sincos)");
      cli |= lyra::opt(options.dataset_size, "value")["--size"]("The size of the dataset (default: 1000)");
      cli |= lyra::opt(options.normalize_data)["--normalize"]("Normalize the data");
      cli |= lyra::opt(options.epochs, "value")["--epochs"]("The number of epochs (default: 100)");
      cli |= lyra::opt(options.batch_size, "value")["--batch-size"]("The batch size of the training algorithm");
      cli |= lyra::opt(options.learning_rate_scheduler, "value")["--learning-rate"]("The learning rate scheduler (default: constant(0.0001))");
      cli |= lyra::opt(options.weights_initialization, "value")["--weights"]("The weight initialization (default, he, uniform, xavier, normalized_xavier, uniform)");
      cli |= lyra::opt(options.loss_function, "value")["--loss"]("The loss function (squared-error, cross-entropy, logistic-cross-entropy)");
      cli |= lyra::opt(options.architecture, "value")["--architecture"]("The architecture of the multilayer perceptron e.g. RRL,\nwhere R=ReLU, S=Sigmoid, L=Linear, B=Batchnorm, Z=Softmax");
      cli |= lyra::opt(hidden_layer_sizes_text, "value")["--hidden"]("A comma separated list of the hidden layer sizes");
      cli |= lyra::opt(options.dropout, "value")["--dropout"]("The dropout rate for the weights of the layers");
      cli |= lyra::opt(options.density, "value")["--density"]("The density rate of the sparse layers");
      cli |= lyra::opt(densities_text, "value")["--densities"]("A comma separated list of sparse layer densities");
      cli |= lyra::opt(options.augmented)["--augmented"]("Load an augmented dataset in every epoch. They should be named epoch<nnn>.npy and stored in datadir");
      cli |= lyra::opt(options.optimizer, "value")["--optimizer"]("The optimizer (gradient-descent, momentum(<mu>), nesterov(<mu>))");
      cli |= lyra::opt(options.seed, "value")["--seed"]("A seed value for the random generator.");
      cli |= lyra::opt(no_shuffle)["--no-shuffle"]("Do not shuffle the dataset during training.");
      cli |= lyra::opt(no_statistics)["--no-statistics"]("Do not compute statistics during training.");
      cli |= lyra::opt(options.threads, "value")["--threads"]("The number of threads used by Eigen.");
      cli |= lyra::opt(datadir, "value")["--datadir"]("A directory containing the files epoch<nnn>.npz");
      // TODO: add import/export of weights in .npz format
      // cli |= lyra::opt(import_weights_file, "value")["--import-weights"]("Loads the weights from a file in .npz format");
      cli |= lyra::opt(export_weights_file, "value")["--export-weights"]("Exports the weights to a file in .npy format");
      cli |= lyra::opt(options.debug)["--debug"]("Show debug output");
      cli |= lyra::opt(options.precision, "value")["--precision"]("The precision that is used for printing.");
      cli |= lyra::opt(options.check_gradients)["--check"]("Check the computed gradients");
      cli |= lyra::opt(options.check_gradients_step, "value")["--gradient-step"]("The step size for approximating the gradient");
      cli |= lyra::opt(info)["--info"]("print some info about the multilayer_perceptron's");
    }

    std::string description() const override
    {
      return "Multilayer perceptron test";
    }

    bool run() override
    {
      if (no_shuffle)
      {
        options.shuffle = false;
      }
      if (no_statistics)
      {
        options.statistics = false;
      }
      std::vector<std::size_t> hidden_layer_sizes = parse_comma_separated_numbers(hidden_layer_sizes_text);
      std::vector<scalar> densities = parse_comma_separated_real_numbers(densities_text);
      check_options(options, hidden_layer_sizes);

      std::mt19937 rng{options.seed};
      NERVA_LOG(log::verbose) << "loading dataset " << options.dataset << std::endl;
      // TODO: loading the dataset should be avoided when the flag --augmented is set.
      // Now this is not possible, because the input and output size are unknown.
      datasets::dataset data = datasets::make_dataset(options.dataset, options.dataset_size, rng);
      std::size_t D = data.Xtrain.rows(); // the number of features
      std::size_t K = data.Ttrain.rows(); // the number of outputs
      NERVA_LOG(log::verbose) << "number of examples: " << data.Xtrain.cols() << std::endl;
      NERVA_LOG(log::verbose) << "number of features: " << D << std::endl;
      NERVA_LOG(log::verbose) << "number of outputs: " << K << std::endl;

      options.sizes = compute_sizes(D, K, hidden_layer_sizes, options.architecture);

      if (densities.empty())
      {
        densities = compute_densities(options.architecture, options.sizes, options.density);
      }
      options.densities = densities;
      options.info();

      std::cout << "number type = " << (std::is_same<scalar, double>::value ? "double" : "float") << "\n\n";

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
          scalar density = densities[densities_index++];
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

      if (import_weights_file.empty())
      {
        set_weights(M, options.weights_initialization, options.architecture, rng);
      }
      else
      {
        import_weights_from_numpy(M, import_weights_file);
      }

      if (!export_weights_file.empty())
      {
        export_weights_to_numpy(M, export_weights_file);
      }

      if (info)
      {
        data.info();
        M.info("before training");
      }

#ifdef NERVA_ENABLE_PROFILING
      CALLGRIND_START_INSTRUMENTATION;
#endif

      std::cout << M.to_string() << std::endl;

      if (options.algorithm == "sgd")
      {
        stochastic_gradient_descent(M, loss, data, options, learning_rate, rng);
      }
      else if (options.algorithm == "minibatch")
      {
        if (options.augmented)
        {
          minibatch_gradient_descent_augmented(M, loss, datadir, options, learning_rate, rng);
        }
        else
        {
          minibatch_gradient_descent(M, loss, data, options, learning_rate, rng);
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

int main(int argc, const char** argv)
{
  pybind11::scoped_interpreter guard{};
  return tool().execute(argc, argv);
}
