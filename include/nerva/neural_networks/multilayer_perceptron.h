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
#include "fmt/format.h"
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
      print_numpy_matrix(name("b"), dlayer->b.transpose());
      index++;
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      print_numpy_matrix(name("W"), mkl::to_eigen(slayer->W));
      print_numpy_matrix(name("b"), slayer->b.transpose());
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
      auto n = slayer->W.values.size();
      auto N = slayer->W.rows() * slayer->W.cols();
      v.push_back(fmt::format("{}/{} ({:.3f}%)", n, N, (100.0 * n) / N));
    }
  }
  return fmt::format("{}", fmt::join(v, ", "));
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

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void export_weights_to_npz(const multilayer_perceptron& M, const std::string& filename)
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
      eigen::matrix W = dlayer->W.transpose();
      eigen::vector b = dlayer->b.reshaped().transpose();
      data[name("W").c_str()] = py::array_t<scalar, py::array::f_style>({W.rows(), W.cols()}, W.data());
      data[name("b").c_str()] = py::array_t<scalar, py::array::f_style>(b.size(), b.data());
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      eigen::matrix W = mkl::to_eigen(slayer->W).transpose();
      eigen::vector b = dlayer->b.reshaped().transpose();
      data[name("W").c_str()] = py::array_t<scalar, py::array::f_style>({W.rows(), W.cols()}, W.data());
      data[name("b").c_str()] = py::array_t<scalar, py::array::f_style>(b.size(), b.data());
    }
  }

  py::module::import("numpy").attr("savez_compressed")(filename, data);
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void import_weights_from_npz(multilayer_perceptron& M, const std::string& filename)
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
      dlayer->import_weights(eigen::extract_matrix<scalar>(data, name("W")));
      dlayer->b = eigen::extract_vector<scalar>(data, name("b")).transpose();
      index++;
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      slayer->import_weights(eigen::extract_matrix<scalar>(data, name("W")));
      slayer->b = eigen::extract_vector<scalar>(data, name("b")).transpose();
      index++;
    }
  }
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

template <typename EigenVector, typename T>
EigenVector convert_to_eigen(const std::vector<T>& x)
{
  typedef typename EigenVector::Scalar Scalar;

  unsigned int size = x.size();
  EigenVector result(size);
  for (unsigned int i = 0; i < size; i++)
  {
    result[i] = static_cast<Scalar>(x[i]);
  }
  return result;
}

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
      np.attr("save")(file, py::array_t<scalar, py::array::f_style>({W.rows(), W.cols()}, W.data()));
    }
    else if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
    {
      const auto& W = slayer->W;
      np.attr("save")(file, py::array_t<scalar>(W.values.size(), W.values.data()));
      np.attr("save")(file, py::array_t<MKL_INT>(W.columns.size(), W.columns.data()));
      np.attr("save")(file, py::array_t<MKL_INT>(W.row_index.size(), W.row_index.data()));
    }
  }
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_MULTILAYER_PERCEPTRON_H
