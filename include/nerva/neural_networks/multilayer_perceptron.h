// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/multilayer_perceptron.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MULTILAYER_PERCEPTRON_H
#define NERVA_NEURAL_NETWORKS_MULTILAYER_PERCEPTRON_H

#include "nerva/neural_networks/check_gradients.h"
#include "nerva/neural_networks/dropout_layers.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/masking.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/numpy_eigen.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/regrowth.h"
#include "nerva/neural_networks/weights.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/stopwatch.h"
#include <pybind11/embed.h>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>

namespace nerva {

struct multilayer_perceptron
{
  std::vector<std::shared_ptr<neural_network_layer>> layers;

  [[nodiscard]] std::string to_string() const
  {
    std::ostringstream out;
    for (const auto& layer: layers)
    {
      out << layer->to_string() << '\n';
    }
    return out.str();
  }

  void feedforward(eigen::matrix& result)
  {
    for (std::size_t i = 0; i < layers.size() - 1; i++)
    {
      layers[i]->feedforward(layers[i+1]->X);
    }
    layers.back()->feedforward(result);
  }

  void feedforward(const eigen::matrix& X, eigen::matrix& result)
  {
#ifdef NERVA_TIMING
    utilities::stopwatch watch;
#endif
    layers.front()->X = X;
    feedforward(result);
#ifdef NERVA_TIMING
    auto seconds = watch.seconds(); std::cout << "feedforward " << seconds << std::endl;
#endif
  }

  void backpropagate(const eigen::matrix& Y, const eigen::matrix& DY)
  {
#ifdef NERVA_TIMING
    utilities::stopwatch watch;
#endif
    layers.back()->backpropagate(Y, DY);
    for (int i = layers.size() - 2; i >= 0; i--)
    {
      layers[i]->backpropagate(layers[i + 1]->X, layers[i + 1]->DX);
    }
#ifdef NERVA_TIMING
    auto seconds = watch.seconds(); std::cout << "backpropagate " << seconds << std::endl;
#endif
  }

  void renew_dropout_mask(std::mt19937& rng)
  {
    for (auto& layer: layers)
    {
      auto dlayer = dynamic_cast<dropout_layer<eigen::matrix>*>(layer.get());
      if (dlayer)
      {
        dlayer->renew(rng);
      }
    }
  }

  void optimize(scalar eta)
  {
    for (auto& layer: layers)
    {
      layer->optimize(eta);
    }
  }

  void regrow(scalar zeta, weight_initialization w, bool separate_positive_negative, std::mt19937& rng)
  {
    for (auto& layer: layers)
    {
      auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get());
      if (slayer)
      {
        nerva::regrow(slayer->W, zeta, w, separate_positive_negative, rng);
      }
    }
  }

  void check_gradients(const std::shared_ptr<loss_function>& loss, const eigen::matrix& T, scalar h = 0.01)
  {
    long K = T.rows();
    long N = T.cols();
    eigen::matrix Y(K, N);

    auto f = [&]()
    {
      feedforward(Y);
      return loss->value(Y, T);
    };

    for (std::size_t i = 0; i < layers.size(); i++)
    {
      // check dense linear layers
      auto layer = dynamic_cast<dense_linear_layer*>(layers[i].get());
      if (layer)
      {
        check_gradient("Db" + std::to_string(i+1), f, layer->b, layer->Db, h);
        check_gradient("DW" + std::to_string(i+1), f, layer->W, layer->DW, h);
      }

      // check sparse mkl linear layers
      auto slayer = dynamic_cast<sparse_linear_layer*>(layers[i].get());
      if (slayer)
      {
        check_gradient("Db" + std::to_string(i+1), f, slayer->b, slayer->Db, h);
        check_gradient("DW" + std::to_string(i+1), f, slayer->W, slayer->DW, h);
      }
    }
  }

  void info(const std::string& name) const
  {
    std::cout << "==================================\n";
    std::cout << " MLP " << name << "\n";
    std::cout << "==================================\n";
    for (unsigned int i = 0; i < layers.size(); i++)
    {
      layers[i]->info(i + 1);
    }
  }
};

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void export_weights_to_numpy(const multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "exporting weights in '.npy' format to file " << filename << std::endl;

  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "wb");

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      np.attr("save")(file, eigen::to_numpy(dlayer->W));
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      np.attr("save")(file, eigen::to_numpy(mkl::to_eigen(slayer->W)));
    }
  }

  file.attr("close")();
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void import_weights_from_numpy(multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "importing weights in '.npy' format from file " << filename << std::endl;

  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "rb");

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      dlayer->W = eigen::from_numpy(np.attr("load")(file).cast<py::array_t<scalar>>());
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      slayer->W = mkl::to_csr(eigen::from_numpy(np.attr("load")(file).cast<py::array_t<scalar>>()));
    }
  }

  file.attr("close")();
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void export_weights_and_bias_to_npz(const multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "Saving weights and bias to file " << filename << std::endl;

  auto np = py::module::import("numpy");
  auto io = py::module::import("io");

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
      data[name("W").c_str()] = eigen::to_numpy(dlayer->W);
      data[name("b").c_str()] = eigen::to_numpy(dlayer->b);
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      data[name("W").c_str()] = eigen::to_numpy(mkl::to_eigen(slayer->W));
      data[name("b").c_str()] = eigen::to_numpy(slayer->b);
    }
  }

  np.attr("savez")(filename, **data);
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void import_weights_and_bias_from_npz(multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "Loading weights and bias from file " << filename << std::endl;

  auto np = py::module::import("numpy");
  auto io = py::module::import("io");

  py::dict data = np.attr("load")(filename);
  unsigned int index = 1;

  auto name = [&](const std::string& name)
  {
    return name + std::to_string(index);
  };

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      dlayer->W = eigen::from_numpy(data[name("W").c_str()].cast<py::array_t<scalar>>()).transpose();
      dlayer->b = eigen::from_numpy(data[name("b").c_str()].cast<py::array_t<scalar>>()).transpose();
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      eigen::matrix W = eigen::from_numpy(data[name("W").c_str()].cast<py::array_t<scalar>>()).transpose();
      slayer->W = mkl::to_csr(W);
      slayer->b = eigen::from_numpy(data[name("b").c_str()].cast<py::array_t<scalar>>()).transpose();
    }
  }
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void export_bias_to_numpy(const multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "exporting bias in '.npy' format to file " << filename << std::endl;

  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "wb");

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      np.attr("save")(file, eigen::to_numpy(dlayer->b));
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      np.attr("save")(file, eigen::to_numpy(slayer->b));
    }
  }

  file.attr("close")();
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void import_bias_from_numpy(multilayer_perceptron& M, const std::string& filename)
{
  namespace py = pybind11;
  NERVA_LOG(log::verbose) << "importing weights in '.npy' format from file " << filename << std::endl;

  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "rb");

  for (auto& layer: M.layers)
  {
    if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
    {
      dlayer->b = eigen::from_numpy(np.attr("load")(file).cast<py::array_t<scalar>>());
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      slayer->b = eigen::from_numpy(np.attr("load")(file).cast<py::array_t<scalar>>());
    }
  }

  file.attr("close")();
}

class mlp_masking
{
  using boolean_matrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

  protected:
    std::vector<boolean_matrix> masks;

  public:
    explicit mlp_masking(const multilayer_perceptron& M)
    {
      // create a binary mask for every sparse linear layer in M
      for (auto& layer: M.layers)
      {
        if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
        {
          masks.push_back(create_binary_mask(mkl::to_eigen(slayer->W)));
        }
      }
    }

    // applies the masks to the dense linear layers in M
    void apply(multilayer_perceptron& M)
    {
      int unsigned index = 0;
      for (auto& layer: M.layers)
      {
        if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
        {
          apply_binary_mask(dlayer->W, masks[index++]);
          if (index >= masks.size())
          {
            break;
          }
        }
      }
    }
};

inline
std::vector<std::pair<long, long>> sparse_linear_layer_sizes(multilayer_perceptron& M)
{
  std::vector<std::pair<long, long>> result;
  for (auto& layer: M.layers)
  {
    if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      const auto& W = slayer->W;
      result.emplace_back(W.rows(), W.cols());
    }
  }
  return result;
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_MULTILAYER_PERCEPTRON_H
