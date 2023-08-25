// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/sgd_options.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/print.h"
#include <iostream>
#include <random>
#include <string>

namespace nerva {

// options for SGD algorithms
struct sgd_options
{
  std::size_t epochs = 100;
  long batch_size = 1;
  bool shuffle = true;
  scalar regrow_rate = 0.0;
  bool regrow_separate_positive_negative = false; // apply the regrow rate to positive and negative values separately
  bool statistics = true;
  bool debug = false;
  scalar gradient_step = 0;  // if gradient_step > 0 then gradient checks will be done

  void info() const;
};

struct mlp_options: public sgd_options
{
  std::string dataset;
  std::size_t dataset_size = 2000;
  bool normalize_data = false;
  std::string learning_rate_scheduler = "constant(0.0001)";
  std::string loss_function = "squared-error";
  std::string architecture;
  std::vector<std::size_t> sizes;
  std::string weights_initialization;
  std::string optimizer = "gradient-descent";
  scalar overall_density = 1;
  std::vector<double> densities;
  std::size_t seed = std::random_device{}();
  int precision = 4;
  int threads = 1;

  void info() const;
};

inline
std::ostream& operator<<(std::ostream& out, const sgd_options& options)
{
  out << "epochs = " << options.epochs << std::endl;
  out << "batch size = " << options.batch_size << std::endl;
  out << "shuffle = " << std::boolalpha << options.shuffle << std::endl;
  if (options.regrow_rate > 0)
  {
    out << "regrow rate = " << options.regrow_rate << std::endl;
    out << "regrow separate positive/negative weights = " << options.regrow_separate_positive_negative << std::endl;
  }
  out << "statistics = " << std::boolalpha << options.statistics << std::endl;
  out << "debug = " << std::boolalpha << options.debug << std::endl;
  return out;
}

inline
std::ostream& operator<<(std::ostream& out, const mlp_options& options)
{
  out << static_cast<const sgd_options&>(options);
  if (!options.dataset.empty())
  {
    out << "dataset = " << options.dataset << std::endl;
    out << "dataset size = " << options.dataset_size << std::endl;
    out << "normalize data = " << std::boolalpha << options.normalize_data << std::endl;
  }
  out << "learning rate scheduler = " << options.learning_rate_scheduler << std::endl;
  out << "loss function = " << options.loss_function << std::endl;
  out << "architecture = " << options.architecture << std::endl;
  out << "sizes = " << print_list(options.sizes) << std::endl;
  out << "weights initialization = " << options.weights_initialization << std::endl;
  out << "optimizer = " << options.optimizer << std::endl;
  out << "overall density = " << options.overall_density << std::endl;
  out << "densities = " << print_list(options.densities) << std::endl;
  out << "seed = " << options.seed << std::endl;
  out << "precision = " << options.precision << std::endl;
  out << "threads = " << options.threads << std::endl;
  return out;
}

inline
void sgd_options::info() const
{
  std::cout << *this;
}

inline
void mlp_options::info() const
{
  std::cout << *this;
}

} // namespace nerva

