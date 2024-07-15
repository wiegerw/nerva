// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mlp_algorithms.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/dropout_layers.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/numpy_eigen.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>

namespace nerva {

inline
void print_model_info(const multilayer_perceptron& M)
{
  unsigned int index = 1;

  auto name = [&](const std::string& name)
  {
    return name + std::to_string(index);
  };

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      print_numpy_matrix(name("W"), dlayer->W);
      print_numpy_matrix(name("b"), dlayer->b);
      index++;
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      print_numpy_matrix(name("W"), mkl::to_eigen(slayer->W));
      print_numpy_matrix(name("b"), slayer->b);
      index++;
    }
    else if (auto blayer = dynamic_cast<batch_normalization_layer*>(layer.get()))
    {
      print_numpy_matrix(name("beta"), blayer->beta);
      print_numpy_matrix(name("gamma"), blayer->gamma);
      index++;
    }
  }
}

inline
std::string layer_density_info(const multilayer_perceptron& M)
{
  std::vector<std::string> v;
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      auto N = dlayer->W.size();
      v.push_back(fmt::format("{}/{} (100%)", N, N));
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      auto n = slayer->W.values().size();
      auto N = slayer->W.rows() * slayer->W.cols();
      v.push_back(fmt::format("{}/{} ({:.3f}%)", n, N, (100.0 * n) / N));
    }
  }
  return fmt::format("{}", fmt::join(v, ", "));
}

inline
void set_support_random(multilayer_perceptron& M, const std::vector<double>& layer_densities, std::mt19937& rng)
{
  std::size_t index = 0;
  for (auto& layer: M.layers)
  {
    if (dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      index++;
    }
    if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      set_support_random(*slayer, layer_densities[index++], rng);
    }
  }
}

inline
void set_weights_and_bias(multilayer_perceptron& M, const std::vector<weight_initialization>& weights, std::mt19937& rng)
{
  unsigned int index = 0;
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      set_weights_and_bias(*dlayer, weights[index++], rng);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      set_weights_and_bias(*slayer, weights[index++], rng);
    }
  }
}

inline
void renew_dropout_masks(multilayer_perceptron& M, std::mt19937& rng)
{
  for (auto& layer: M.layers)
  {
    auto dlayer = dynamic_cast<dropout_layer<eigen::matrix>*>(layer.get());
    if (dlayer)
    {
      dlayer->renew(rng);
    }
  }
}

inline
std::vector<eigen::matrix> mlp_weights(const multilayer_perceptron& M)
{
  std::vector<eigen::matrix> result;
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      result.push_back(dlayer->W);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      result.push_back(mkl::to_eigen(slayer->W));
    }
  }
  return result;
}

inline
std::vector<eigen::matrix> mlp_bias(const multilayer_perceptron& M)
{
  std::vector<eigen::matrix> result;
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      result.push_back(dlayer->b);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      result.push_back(slayer->b);
    }
  }
  return result;
}

inline
bool has_nan(const multilayer_perceptron& M)
{
  std::vector<eigen::matrix> result;
  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      if (nerva::has_nan(dlayer->W))
      {
        return true;
      }
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      if (mkl::has_nan(slayer->W))
      {
        return true;
      }
    }
  }
  return false;
}

inline
void print_srelu_layers(multilayer_perceptron& M)
{
  unsigned int index = 1;

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_srelu_layer*>(layer.get()))
    {
      std::cout << "layer " << index << ' ' << dlayer->act.to_string() << std::endl;
      index++;
    }
    else if (auto slayer = dynamic_cast<sparse_srelu_layer*>(layer.get()))
    {
      std::cout << "layer " << index << ' ' << slayer->act.to_string() << std::endl;
      index++;
    }
    else if (auto player = dynamic_cast<dense_srelu_dropout_layer*>(layer.get()))
    {
      std::cout << "layer " << index << ' ' << player->act.to_string() << std::endl;
      index++;
    }
  }
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void save_weights_and_bias(const multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "Saving weights and bias to file " << filename << std::endl;

  py::dict data;
  unsigned int index = 1;

  auto name = [&](const std::string& name)
  {
    return name + std::to_string(index);
  };

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      eigen::matrix W = dlayer->W;
      eigen::matrix b = dlayer->b;
      data[name("W").c_str()] = pybind11::array_t<scalar, py::array::f_style>({W.rows(), W.cols()}, W.data());
      data[name("b").c_str()] = pybind11::array_t<scalar, py::array::f_style>(b.size(), b.data());
      index++;
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      eigen::matrix W = mkl::to_eigen(slayer->W);
      eigen::matrix b = slayer->b;
      data[name("W").c_str()] = pybind11::array_t<scalar, py::array::f_style>({W.rows(), W.cols()}, W.data());
      data[name("b").c_str()] = pybind11::array_t<scalar, py::array::f_style>(b.size(), b.data());
      index++;
    }
  }

  py::module::import("numpy").attr("savez_compressed")(filename, **data);
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void load_weights_and_bias(multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "Loading weights and bias from file " << filename << std::endl;

  py::dict data = py::module::import("numpy").attr("load")(filename);
  unsigned int index = 1;

  auto name = [&](const std::string& name)
  {
    return name + std::to_string(index);
  };

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      dlayer->load_weights(eigen::extract_matrix<scalar>(data, name("W")));
      dlayer->b = eigen::extract_row_vector<scalar>(data, name("b"));
      index++;
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      slayer->load_weights(mkl::to_csr(eigen::extract_matrix<scalar>(data, name("W"))));
      slayer->b = eigen::extract_row_vector<scalar>(data, name("b"));
      index++;
    }
  }
}

// This function can be used to get a rough indication of the size of a model.
inline
void save_model_weights_to_npy(const std::string& filename, const multilayer_perceptron& M)
{
  std::cout << "Saving model weights to " << filename << std::endl;

  namespace py = pybind11;
  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "wb");

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      const auto& W = dlayer->W;
      np.attr("save")(file, pybind11::array_t<scalar, py::array::f_style>({W.rows(), W.cols()}, W.data()));
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      const auto& W = slayer->W;
      np.attr("save")(file, pybind11::array_t<scalar>(W.values().size(), W.values().data()));
      np.attr("save")(file, py::array_t<MKL_INT>(W.col_index().size(), W.col_index().data()));
      np.attr("save")(file, py::array_t<MKL_INT>(W.row_index().size(), W.row_index().data()));
    }
  }
}

} // namespace nerva
