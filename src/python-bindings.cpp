// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file src/python-bindings.cpp
/// \brief add your file description here.

#include "nerva/datasets/dataset.h"
#include "nerva/neural_networks/activation_functions.h"
#include "nerva/neural_networks/batch_normalization_layers.h"
#include "nerva/neural_networks/dropout_layers.h"
#include "nerva/neural_networks/nerva_timer.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/loss_functions.h"
#include "nerva/neural_networks/masking.h"
#include "nerva/neural_networks/mlp_algorithms.h"
#include "nerva/neural_networks/parse_layer.h"
#include "nerva/neural_networks/random.h"
#include "nerva/neural_networks/regrow.h"
#include "nerva/neural_networks/softmax_functions.h"
#include "nerva/neural_networks/training.h"
#include "nerva/neural_networks/weights.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace nerva;

PYBIND11_MODULE(nervalibcolwise, m)
{
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  /////////////////////////////////////////////////////////////////////////
  //                       utilities
  /////////////////////////////////////////////////////////////////////////

  py::class_<std::mt19937>(m, "RandomNumberGenerator")
    .def(py::init<std::uint32_t>(), py::return_value_policy::copy)
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       datasets
  /////////////////////////////////////////////////////////////////////////

  py::class_<datasets::dataset, std::shared_ptr<datasets::dataset>>(m, "data_set")
    .def(py::init<>(), py::return_value_policy::copy)
    .def("info", &datasets::dataset::info)
    .def("load", &datasets::dataset::load)
    .def("save", &datasets::dataset::save)
    ;

  py::class_<datasets::dataset_view, std::shared_ptr<datasets::dataset_view>>(m, "DataSetView")
    .def(py::init<datasets::matrix_ref, datasets::matrix_ref, datasets::matrix_ref, datasets::matrix_ref>(), py::return_value_policy::copy)
    .def("info", &datasets::dataset_view::info)
    .def("save", &datasets::dataset_view::save)
    .def_readwrite("Xtrain", &datasets::dataset_view::Xtrain, py::return_value_policy::reference_internal)
    .def_readwrite("Ttrain", &datasets::dataset_view::Ttrain, py::return_value_policy::reference_internal)
    .def_readwrite("Xtest", &datasets::dataset_view::Xtest, py::return_value_policy::reference_internal)
    .def_readwrite("Ttest", &datasets::dataset_view::Ttest, py::return_value_policy::reference_internal)
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       loss functions
  /////////////////////////////////////////////////////////////////////////

  py::class_<loss_function, std::shared_ptr<loss_function>>(m, "loss_function")
    .def("value", &loss_function::value)
    .def("gradient", &loss_function::gradient)
    ;

  py::class_<squared_error_loss, loss_function, std::shared_ptr<squared_error_loss>>(m, "squared_error_loss")
    .def(py::init<>(), py::return_value_policy::copy)
    ;

  py::class_<cross_entropy_loss, loss_function, std::shared_ptr<cross_entropy_loss>>(m, "cross_entropy_loss")
    .def(py::init<>(), py::return_value_policy::copy)
    ;

  py::class_<logistic_cross_entropy_loss, loss_function, std::shared_ptr<logistic_cross_entropy_loss>>(m, "logistic_cross_entropy_loss")
    .def(py::init<>(), py::return_value_policy::copy)
    ;

  py::class_<softmax_cross_entropy_loss, loss_function, std::shared_ptr<softmax_cross_entropy_loss>>(m, "softmax_cross_entropy_loss")
    .def(py::init<>(), py::return_value_policy::copy)
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       learning rate schedulers
  /////////////////////////////////////////////////////////////////////////

  py::class_<learning_rate_scheduler, std::shared_ptr<learning_rate_scheduler>>(m, "learning_rate_scheduler")
    .def("__call__", &learning_rate_scheduler::operator())
    ;

  py::class_<constant_scheduler, learning_rate_scheduler, std::shared_ptr<constant_scheduler>>(m, "constant_scheduler")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def_readwrite("lr", &constant_scheduler::lr, py::return_value_policy::reference_internal)
    ;

  py::class_<time_based_scheduler, learning_rate_scheduler, std::shared_ptr<time_based_scheduler>>(m, "time_based_scheduler")
    .def(py::init<scalar, scalar>(), py::return_value_policy::copy)
    .def_readwrite("lr", &time_based_scheduler::lr, py::return_value_policy::reference_internal)
    .def_readwrite("decay", &time_based_scheduler::decay, py::return_value_policy::reference_internal)
    ;

  py::class_<step_based_scheduler, learning_rate_scheduler, std::shared_ptr<step_based_scheduler>>(m, "step_based_scheduler")
    .def(py::init<scalar, scalar, scalar>(), py::return_value_policy::copy)
    .def_readwrite("lr", &step_based_scheduler::lr, py::return_value_policy::reference_internal)
    .def_readwrite("drop_rate", &step_based_scheduler::drop_rate, py::return_value_policy::reference_internal)
    .def_readwrite("change_rate", &step_based_scheduler::change_rate, py::return_value_policy::reference_internal)
    ;

  py::class_<multi_step_lr_scheduler, learning_rate_scheduler, std::shared_ptr<multi_step_lr_scheduler>>(m, "multi_step_lr_scheduler")
    .def(py::init<scalar, std::vector<unsigned int>, scalar>(), py::return_value_policy::copy)
    .def_readwrite("lr", &multi_step_lr_scheduler::lr, py::return_value_policy::reference_internal)
    .def_readwrite("milestones", &multi_step_lr_scheduler::milestones, py::return_value_policy::reference_internal)
    .def_readwrite("gamma", &multi_step_lr_scheduler::gamma, py::return_value_policy::reference_internal)
    ;

  py::class_<exponential_scheduler, learning_rate_scheduler, std::shared_ptr<exponential_scheduler>>(m, "exponential_scheduler")
    .def(py::init<scalar, scalar>(), py::return_value_policy::copy)
    .def_readwrite("lr", &exponential_scheduler::lr, py::return_value_policy::reference_internal)
    .def_readwrite("change_rate", &exponential_scheduler::change_rate, py::return_value_policy::reference_internal)
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       optimizers
  /////////////////////////////////////////////////////////////////////////

  py::class_<optimizer_function, std::shared_ptr<optimizer_function>>(m, "layer_optimizer")
    .def("update", &optimizer_function::update)
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       layers
  /////////////////////////////////////////////////////////////////////////

  //--- dense layers ---//
  py::class_<neural_network_layer, std::shared_ptr<neural_network_layer>>(m, "neural_network_layer")
    .def_readwrite("X", &dense_linear_layer::X)
    .def_readwrite("DX", &dense_linear_layer::DX)
    .def("info", &neural_network_layer::info)
    ;

  py::class_<dense_linear_layer, neural_network_layer, std::shared_ptr<dense_linear_layer>>(m, "linear_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    .def_readwrite("W", &dense_linear_layer::W)
    .def_readwrite("DW", &dense_linear_layer::DW)
    .def_readwrite("b", &dense_linear_layer::b)
    .def_readwrite("Db", &dense_linear_layer::Db)
    .def_readwrite("optimizer", &dense_linear_layer::optimizer)
    .def("input_size", &dense_linear_layer::input_size)
    .def("output_size", &dense_linear_layer::output_size)
    .def("feedforward", &dense_linear_layer::feedforward)
    .def("backpropagate", &dense_linear_layer::backpropagate)
    .def("optimize", &dense_linear_layer::optimize)
    .def("initialize_weights", [](dense_linear_layer& layer, weight_initialization w) { initialize_weights(w, layer.W, layer.b, nerva_rng); })
    .def("set_linear_layer_optimizer", [](dense_linear_layer& layer, const std::string& text) { set_linear_layer_optimizer<eigen::matrix>(layer, text); })
    .def("set_weights_and_bias", [](dense_linear_layer& layer, weight_initialization w) { set_weights_and_bias(layer, w, nerva_rng); })
    ;

  py::class_<dense_relu_layer, dense_linear_layer, std::shared_ptr<dense_relu_layer>>(m, "relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_trelu_layer, dense_linear_layer, std::shared_ptr<dense_trelu_layer>>(m, "trimmed_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_all_relu_layer, dense_linear_layer, std::shared_ptr<dense_all_relu_layer>>(m, "all_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_leaky_relu_layer, dense_linear_layer, std::shared_ptr<dense_leaky_relu_layer>>(m, "leaky_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_sigmoid_layer, dense_linear_layer, std::shared_ptr<dense_sigmoid_layer>>(m, "sigmoid_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_softmax_layer, dense_linear_layer, std::shared_ptr<dense_softmax_layer>>(m, "softmax_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_log_softmax_layer, dense_linear_layer, std::shared_ptr<dense_log_softmax_layer>>(m, "log_softmax_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_hyperbolic_tangent_layer, dense_linear_layer, std::shared_ptr<dense_hyperbolic_tangent_layer>>(m, "hyperbolic_tangent_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  //--- dropout layers ---//
  py::class_<dense_linear_dropout_layer, dense_linear_layer, std::shared_ptr<dense_linear_dropout_layer>>(m, "linear_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    .def("initialize_weights", [](dense_linear_dropout_layer& layer, weight_initialization w) { initialize_weights(w, layer.W, layer.b, nerva_rng); })
    .def("set_linear_layer_optimizer", [](dense_linear_dropout_layer& layer, const std::string& text) {
      set_linear_layer_optimizer(layer, text); })
    ;

  py::class_<dense_relu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_relu_dropout_layer>>(m, "relu_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_all_relu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_all_relu_dropout_layer>>(m, "all_relu_dropout_layer")
    .def(py::init<scalar, std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_leaky_relu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_leaky_relu_dropout_layer>>(m, "leaky_relu_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_sigmoid_dropout_layer, dense_linear_layer, std::shared_ptr<dense_sigmoid_dropout_layer>>(m, "sigmoid_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_softmax_dropout_layer, dense_linear_layer, std::shared_ptr<dense_softmax_dropout_layer>>(m, "softmax_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_hyperbolic_tangent_dropout_layer, dense_linear_layer, std::shared_ptr<dense_hyperbolic_tangent_dropout_layer>>(m, "hyperbolic_tangent_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_trelu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_trelu_dropout_layer>>(m, "trelu_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_srelu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_srelu_dropout_layer>>(m, "srelu_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar, scalar, scalar, scalar, scalar>(), py::return_value_policy::copy)
    ;

  //--- sparse layers ---//
  py::class_<mkl::sparse_matrix_csr<scalar>, std::shared_ptr<mkl::sparse_matrix_csr<scalar>>>(m, "sparse_matrix_csr")
    .def(py::init<>(), py::return_value_policy::copy)
    .def("rows", &mkl::sparse_matrix_csr<scalar>::rows)
    .def("cols", &mkl::sparse_matrix_csr<scalar>::cols)
    .def("nonzero_count", [](const mkl::sparse_matrix_csr<scalar>& x) { return std::make_pair<long, long>(static_cast<long>(x.values().size()), x.rows() * x.cols()); })
    //.def_readwrite("values", &mkl::sparse_matrix_csr<scalar>::m_values)  TODO: how to get access to protected members?
    //.def_readwrite("columns", &mkl::sparse_matrix_csr<scalar>::m_col_index)
    //.def_readwrite("row_index", &mkl::sparse_matrix_csr<scalar>::m_row_index)
    ;

  py::class_<sparse_linear_layer, neural_network_layer, std::shared_ptr<linear_layer<mkl::sparse_matrix_csr<scalar>>>>(m, "sparse_linear_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    .def_readwrite("W", &sparse_linear_layer::W)
    .def_readwrite("DW", &sparse_linear_layer::DW)
    .def_readwrite("b", &sparse_linear_layer::b)
    .def_readwrite("Db", &sparse_linear_layer::Db)
    .def_readwrite("optimizer", &sparse_linear_layer::optimizer)
    .def("input_size", &sparse_linear_layer::input_size)
    .def("output_size", &sparse_linear_layer::output_size)
    .def("feedforward", &sparse_linear_layer::feedforward)
    .def("backpropagate", &sparse_linear_layer::backpropagate)
    .def("optimize", &sparse_linear_layer::optimize)
    .def("initialize_weights", [](sparse_linear_layer& layer, weight_initialization w) { initialize_weights(w, layer.W, layer.b, nerva_rng); })
    .def("set_linear_layer_optimizer", [](sparse_linear_layer& layer, const std::string& text) { set_linear_layer_optimizer(layer, text); })
    .def("shape", [](sparse_linear_layer& layer) { return std::make_pair(layer.W.rows(), layer.W.cols()); })
    .def("weight_count", [](sparse_linear_layer& layer) { return support_size(layer.W); })
    .def("positive_weight_count", [](sparse_linear_layer& layer) { return count_positive_elements(layer.W); })
    .def("negative_weight_count", [](sparse_linear_layer& layer) { return count_negative_elements(layer.W); })
    .def("set_support_random", [](sparse_linear_layer& layer, scalar density) { set_support_random(layer, density, nerva_rng); })
    .def("set_weights_and_bias", [](sparse_linear_layer& layer, weight_initialization w) { set_weights_and_bias(layer, w, nerva_rng); })
    .def("prune_magnitude", [](sparse_linear_layer& layer, scalar zeta)
    {
      std::size_t count = std::lround(zeta * static_cast<scalar>(mkl::support_size(layer.W)));
      return prune_magnitude(layer.W, count, std::numeric_limits<scalar>::quiet_NaN());
    })
    .def("prune_threshold", [](sparse_linear_layer& layer, scalar threshold)
    {
      return prune_threshold(layer.W, threshold, std::numeric_limits<scalar>::quiet_NaN());
    })
    .def("prune_SET", [](sparse_linear_layer& layer, scalar zeta)
    {
      return prune_SET(layer.W, zeta, std::numeric_limits<scalar>::quiet_NaN());
    })
    .def("grow_random", [](sparse_linear_layer& layer, const std::string& init, std::size_t count)
    {
      weight_initialization w = parse_weight_initialization(init);
      grow_random(layer.W, make_weight_initializer(w, layer.W, nerva_rng), count, nerva_rng);
      layer.reset_support();
    })
    ;

  py::class_<sparse_hyperbolic_tangent_layer, sparse_linear_layer, std::shared_ptr<sparse_hyperbolic_tangent_layer>>(m, "sparse_hyperbolic_tangent_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_relu_layer, sparse_linear_layer, std::shared_ptr<sparse_relu_layer>>(m, "sparse_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_trelu_layer, sparse_linear_layer, std::shared_ptr<sparse_trelu_layer>>(m, "sparse_trimmed_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_leaky_relu_layer, sparse_linear_layer, std::shared_ptr<sparse_leaky_relu_layer>>(m, "sparse_leaky_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_all_relu_layer, sparse_linear_layer, std::shared_ptr<sparse_all_relu_layer>>(m, "sparse_all_relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_sigmoid_layer, sparse_linear_layer, std::shared_ptr<sparse_sigmoid_layer>>(m, "sparse_sigmoid_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_softmax_layer, sparse_linear_layer, std::shared_ptr<sparse_softmax_layer>>(m, "sparse_softmax_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<sparse_log_softmax_layer, sparse_linear_layer, std::shared_ptr<sparse_log_softmax_layer>>(m, "sparse_log_softmax_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  //--- batch normalization layers ---//
  py::class_<dense_batch_normalization_layer, neural_network_layer, std::shared_ptr<dense_batch_normalization_layer>>(m, "batch_normalization_layer")
    .def(py::init<std::size_t, std::size_t>(), py::return_value_policy::copy)
  ;

  py::class_<dense_simple_batch_normalization_layer, neural_network_layer, std::shared_ptr<dense_simple_batch_normalization_layer>>(m, "simple_batch_normalization_layer")
    .def(py::init<std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_affine_layer, neural_network_layer, std::shared_ptr<dense_affine_layer>>(m, "affine_layer")
    .def(py::init<std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  m.def("compute_sparse_layer_densities", compute_sparse_layer_densities);
  m.def("is_linear_layer", is_linear_layer);

  m.def("make_dense_linear_layer", [](std::size_t D,
                                      std::size_t K,
                                      long N,
                                      const std::string& activation,
                                      const std::string& weights,
                                      const std::string& optimizer
  )
  {
    return make_dense_linear_layer(D, K, N, activation, weights, optimizer, nerva_rng);
  });

  m.def("make_dense_linear_dropout_layer", [](std::size_t D,
                                              std::size_t K,
                                              long N,
                                              scalar dropout_rate,
                                              const std::string& activation,
                                              const std::string& weights,
                                              const std::string& optimizer
  )
  {
    return make_dense_linear_dropout_layer(D, K, N, dropout_rate, activation, weights, optimizer, nerva_rng);
  });

  m.def("make_sparse_linear_layer", [](std::size_t D,
                                       std::size_t K,
                                       long N,
                                       scalar density,
                                       const std::string& activation,
                                       const std::string& weights,
                                       const std::string& optimizer
  )
  {
    return make_sparse_linear_layer(D, K, N, density, activation, weights, optimizer, nerva_rng);
  });

  m.def("make_batch_normalization_layer", [](std::size_t D,
                                             long N,
                                             const std::string& optimizer
  )
  {
    return make_batch_normalization_layer(D, N, optimizer);
  });

  /////////////////////////////////////////////////////////////////////////
  //                       multilayer perceptron
  /////////////////////////////////////////////////////////////////////////

  py::class_<multilayer_perceptron, std::shared_ptr<multilayer_perceptron>>(m, "MLP")
    .def(py::init<>(), py::return_value_policy::copy)
    .def("__str__", [](const multilayer_perceptron& M) { return M.to_string(); })
    .def_readwrite("layers", &multilayer_perceptron::layers)
    .def("feedforward", [](multilayer_perceptron& M, const eigen::matrix& X) { eigen::matrix Y; M.feedforward(X, Y); return Y; })
    .def("backpropagate", &multilayer_perceptron::backpropagate)
    .def("optimize", &multilayer_perceptron::optimize)
    .def("clip", &multilayer_perceptron::clip)
    .def("append_layer", [](multilayer_perceptron& M, const std::shared_ptr<neural_network_layer>& layer) { M.layers.push_back(layer); })
    .def("info", &multilayer_perceptron::info)
    .def("save_weights_and_bias", [](const multilayer_perceptron& M, const std::string& filename) { save_weights_and_bias(M, filename); })
    .def("load_weights_and_bias", [](multilayer_perceptron& M, const std::string& filename) { load_weights_and_bias(M, filename); })
    .def("set_weights_and_bias", [](multilayer_perceptron& M, const std::vector<weight_initialization>& weights) { set_weights_and_bias(M, weights, nerva_rng); })
    .def("weights", [](multilayer_perceptron& M) { return mlp_weights(M); })
    .def("bias", [](multilayer_perceptron& M) { return mlp_bias(M); })
    ;

  py::class_<mlp_masking, std::shared_ptr<mlp_masking>>(m, "MLPMasking")
    .def(py::init<const multilayer_perceptron&>(), py::return_value_policy::copy)
    .def("apply", &mlp_masking::apply)
    ;

  m.def("save_model_weights_to_npy", save_model_weights_to_npy);
  m.def("print_model_info", print_model_info);
  m.def("renew_dropout_masks", [](multilayer_perceptron& M) { renew_dropout_masks(M, nerva_rng); });

  /////////////////////////////////////////////////////////////////////////
  //                       weights
  /////////////////////////////////////////////////////////////////////////

  py::enum_<weight_initialization>(m, "Weights")
    .value("He", weight_initialization::he, "He")
    .value("Xavier", weight_initialization::xavier, "Xavier")
    .value("XavierNormalized", weight_initialization::xavier_normalized, "XavierNormalized")
    .value("Uniform", weight_initialization::uniform, "Uniform")
    .value("PyTorch", weight_initialization::pytorch, "PyTorch")
    .value("Zero", weight_initialization::zero, "Zero")
    .value("None_", weight_initialization::none, "None")
    ;

  m.def("initialize_weights", initialize_weights<eigen::matrix>);

  /////////////////////////////////////////////////////////////////////////
  //                       training
  /////////////////////////////////////////////////////////////////////////

  py::class_<sgd_options>(m, "sgd_options")
    .def(py::init<>(), py::return_value_policy::copy)
    .def_readwrite("batch_size", &sgd_options::batch_size)
    .def_readwrite("epochs", &sgd_options::epochs)
    .def_readwrite("debug", &sgd_options::debug)
    .def_readwrite("clip", &sgd_options::clip)
    .def_readwrite("shuffle", &sgd_options::shuffle)
    .def_readwrite("statistics", &sgd_options::statistics)
    .def("info", &sgd_options::info)
    ;

  py::class_<stochastic_gradient_descent_algorithm<datasets::dataset_view>>(m, "stochastic_gradient_descent_algorithm")
    .def(py::init(
         [](multilayer_perceptron& M,
            datasets::dataset_view& data,
            const sgd_options& options,
            const std::shared_ptr<loss_function>& loss,
            const std::shared_ptr<learning_rate_scheduler>& learning_rate
         )
         {
           return new stochastic_gradient_descent_algorithm<datasets::dataset_view>(M, data, options, loss, learning_rate, nerva_rng);
         }
        ))
    .def("run", &stochastic_gradient_descent_algorithm<datasets::dataset_view>::run)
    ;

  m.def("compute_loss", compute_loss);
  m.def("compute_accuracy", compute_accuracy<datasets::matrix_ref>);
  m.def("compute_statistics", compute_statistics<datasets::dataset_view>);
  m.def("set_num_threads", mkl_set_num_threads);
  m.def("set_nerva_computation", set_nerva_computation);

  /////////////////////////////////////////////////////////////////////////
  //                       activation functions
  /////////////////////////////////////////////////////////////////////////

  m.def("Relu", [](const eigen::matrix_ref<scalar>& X) { return Relu(X); });
  m.def("Relu_gradient", [](const eigen::matrix_ref<scalar>& X) { return Relu_gradient(X); });
  m.def("Sigmoid", [](const eigen::matrix_ref<scalar>& X) { return Sigmoid(X); });
  m.def("Sigmoid_gradient", [](const eigen::matrix_ref<scalar>& X) { return Sigmoid_gradient(X); });
  m.def("Hyperbolic_tangent", [](const eigen::matrix_ref<scalar>& X) { return Hyperbolic_tangent(X); });
  m.def("Hyperbolic_tangent_gradient", [](const eigen::matrix_ref<scalar>& X) { return Hyperbolic_tangent_gradient(X); });

  py::class_<Leaky_relu>(m, "Leaky_relu")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def("__call__", [](Leaky_relu& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<Leaky_relu_gradient>(m, "Leaky_relu_gradient")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def("__call__", [](Leaky_relu_gradient& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<All_relu>(m, "All_relu")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def("__call__", [](All_relu& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<All_relu_gradient>(m, "All_relu_gradient")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def("__call__", [](All_relu_gradient& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<Srelu>(m, "Srelu")
    .def(py::init<scalar, scalar, scalar, scalar>(), py::return_value_policy::copy)
    .def("__call__", [](Srelu& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<Srelu_gradient>(m, "Srelu_gradient")
    .def(py::init<scalar, scalar, scalar, scalar>(), py::return_value_policy::copy)
    .def("__call__", [](Srelu_gradient& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<Trimmed_relu>(m, "Trimmed_relu")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def("__call__", [](Trimmed_relu& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  py::class_<Trimmed_relu_gradient>(m, "Trimmed_relu_gradient")
    .def(py::init<scalar>(), py::return_value_policy::copy)
    .def("__call__", [](Trimmed_relu_gradient& f, const eigen::matrix_ref<scalar>& X) { return f(X); })
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       loss functions
  /////////////////////////////////////////////////////////////////////////

  m.def("squared_error_loss_colwise"                                , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return squared_error_loss_colwise(y, t); });
  m.def("squared_error_loss_colwise_gradient"                       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return squared_error_loss_colwise_gradient(y, t); });
  m.def("Squared_error_loss_colwise"                                , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Squared_error_loss_colwise(Y, T); });
  m.def("Squared_error_loss_colwise_gradient"                       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Squared_error_loss_colwise_gradient(Y, T); });
  m.def("mean_squared_error_loss_colwise"                           , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return mean_squared_error_loss_colwise(y, t); });
  m.def("mean_squared_error_loss_colwise_gradient"                  , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return mean_squared_error_loss_colwise_gradient(y, t); });
  m.def("Mean_squared_error_loss_colwise"                           , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Mean_squared_error_loss_colwise(Y, T); });
  m.def("Mean_squared_error_loss_colwise_gradient"                  , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Mean_squared_error_loss_colwise_gradient(Y, T); });
  m.def("cross_entropy_loss_colwise"                                , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return cross_entropy_loss_colwise(y, t); });
  m.def("cross_entropy_loss_colwise_gradient"                       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return cross_entropy_loss_colwise_gradient(y, t); });
  m.def("Cross_entropy_loss_colwise"                                , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Cross_entropy_loss_colwise(Y, T); });
  m.def("Cross_entropy_loss_colwise_gradient"                       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Cross_entropy_loss_colwise_gradient(Y, T); });
  m.def("softmax_cross_entropy_loss_colwise"                        , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return softmax_cross_entropy_loss_colwise(y, t); });
  m.def("softmax_cross_entropy_loss_colwise_gradient"               , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return softmax_cross_entropy_loss_colwise_gradient(y, t); });
  m.def("softmax_cross_entropy_loss_colwise_gradient_one_hot"       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return softmax_cross_entropy_loss_colwise_gradient_one_hot(y, t); });
  m.def("Softmax_cross_entropy_loss_colwise"                        , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Softmax_cross_entropy_loss_colwise(Y, T); });
  m.def("Softmax_cross_entropy_loss_colwise_gradient"               , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Softmax_cross_entropy_loss_colwise_gradient(Y, T); });
  m.def("Softmax_cross_entropy_loss_colwise_gradient_one_hot"       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Softmax_cross_entropy_loss_colwise_gradient_one_hot(Y, T); });
  m.def("stable_softmax_cross_entropy_loss_colwise"                 , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return stable_softmax_cross_entropy_loss_colwise(y, t); });
  m.def("stable_softmax_cross_entropy_loss_colwise_gradient"        , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return stable_softmax_cross_entropy_loss_colwise_gradient(y, t); });
  m.def("stable_softmax_cross_entropy_loss_colwise_gradient_one_hot", [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(y, t); });
  m.def("Stable_softmax_cross_entropy_loss_colwise"                 , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Stable_softmax_cross_entropy_loss_colwise(Y, T); });
  m.def("Stable_softmax_cross_entropy_loss_colwise_gradient"        , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Stable_softmax_cross_entropy_loss_colwise_gradient(Y, T); });
  m.def("Stable_softmax_cross_entropy_loss_colwise_gradient_one_hot", [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(Y, T); });
  m.def("logistic_cross_entropy_loss_colwise"                       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return logistic_cross_entropy_loss_colwise(y, t); });
  m.def("logistic_cross_entropy_loss_colwise_gradient"              , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return logistic_cross_entropy_loss_colwise_gradient(y, t); });
  m.def("Logistic_cross_entropy_loss_colwise"                       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Logistic_cross_entropy_loss_colwise(Y, T); });
  m.def("Logistic_cross_entropy_loss_colwise_gradient"              , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Logistic_cross_entropy_loss_colwise_gradient(Y, T); });
  m.def("negative_log_likelihood_loss_colwise"                      , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return negative_log_likelihood_loss_colwise(y, t); });
  m.def("negative_log_likelihood_loss_colwise_gradient"             , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return negative_log_likelihood_loss_colwise_gradient(y, t); });
  m.def("Negative_log_likelihood_loss_colwise"                      , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Negative_log_likelihood_loss_colwise(Y, T); });
  m.def("Negative_log_likelihood_loss_colwise_gradient"             , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Negative_log_likelihood_loss_colwise_gradient(Y, T); });
  m.def("squared_error_loss_rowwise"                                , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return squared_error_loss_rowwise(y, t); });
  m.def("squared_error_loss_rowwise_gradient"                       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return squared_error_loss_rowwise_gradient(y, t); });
  m.def("Squared_error_loss_rowwise"                                , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Squared_error_loss_rowwise(Y, T); });
  m.def("Squared_error_loss_rowwise_gradient"                       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Squared_error_loss_rowwise_gradient(Y, T); });
  m.def("mean_squared_error_loss_rowwise"                           , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return mean_squared_error_loss_rowwise(y, t); });
  m.def("mean_squared_error_loss_rowwise_gradient"                  , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return mean_squared_error_loss_rowwise_gradient(y, t); });
  m.def("Mean_squared_error_loss_rowwise"                           , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Mean_squared_error_loss_rowwise(Y, T); });
  m.def("Mean_squared_error_loss_rowwise_gradient"                  , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Mean_squared_error_loss_rowwise_gradient(Y, T); });
  m.def("cross_entropy_loss_rowwise"                                , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return cross_entropy_loss_rowwise(y, t); });
  m.def("cross_entropy_loss_rowwise_gradient"                       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return cross_entropy_loss_rowwise_gradient(y, t); });
  m.def("Cross_entropy_loss_rowwise"                                , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Cross_entropy_loss_rowwise(Y, T); });
  m.def("Cross_entropy_loss_rowwise_gradient"                       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Cross_entropy_loss_rowwise_gradient(Y, T); });
  m.def("softmax_cross_entropy_loss_rowwise"                        , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return softmax_cross_entropy_loss_rowwise(y, t); });
  m.def("softmax_cross_entropy_loss_rowwise_gradient"               , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return softmax_cross_entropy_loss_rowwise_gradient(y, t); });
  m.def("softmax_cross_entropy_loss_rowwise_gradient_one_hot"       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return softmax_cross_entropy_loss_rowwise_gradient_one_hot(y, t); });
  m.def("Softmax_cross_entropy_loss_rowwise"                        , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Softmax_cross_entropy_loss_rowwise(Y, T); });
  m.def("Softmax_cross_entropy_loss_rowwise_gradient"               , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Softmax_cross_entropy_loss_rowwise_gradient(Y, T); });
  m.def("Softmax_cross_entropy_loss_rowwise_gradient_one_hot"       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Softmax_cross_entropy_loss_rowwise_gradient_one_hot(Y, T); });
  m.def("stable_softmax_cross_entropy_loss_rowwise"                 , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return stable_softmax_cross_entropy_loss_rowwise(y, t); });
  m.def("stable_softmax_cross_entropy_loss_rowwise_gradient"        , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return stable_softmax_cross_entropy_loss_rowwise_gradient(y, t); });
  m.def("stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot", [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(y, t); });
  m.def("Stable_softmax_cross_entropy_loss_rowwise"                 , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Stable_softmax_cross_entropy_loss_rowwise(Y, T); });
  m.def("Stable_softmax_cross_entropy_loss_rowwise_gradient"        , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Stable_softmax_cross_entropy_loss_rowwise_gradient(Y, T); });
  m.def("Stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot", [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(Y, T); });
  m.def("logistic_cross_entropy_loss_rowwise"                       , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return logistic_cross_entropy_loss_rowwise(y, t); });
  m.def("logistic_cross_entropy_loss_rowwise_gradient"              , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return logistic_cross_entropy_loss_rowwise_gradient(y, t); });
  m.def("Logistic_cross_entropy_loss_rowwise"                       , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Logistic_cross_entropy_loss_rowwise(Y, T); });
  m.def("Logistic_cross_entropy_loss_rowwise_gradient"              , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Logistic_cross_entropy_loss_rowwise_gradient(Y, T); });
  m.def("negative_log_likelihood_loss_rowwise"                      , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return negative_log_likelihood_loss_rowwise(y, t); });
  m.def("negative_log_likelihood_loss_rowwise_gradient"             , [](const eigen::matrix_ref<scalar>& y, const eigen::matrix_ref<scalar>& t) { return negative_log_likelihood_loss_rowwise_gradient(y, t); });
  m.def("Negative_log_likelihood_loss_rowwise"                      , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Negative_log_likelihood_loss_rowwise(Y, T); });
  m.def("Negative_log_likelihood_loss_rowwise_gradient"             , [](const eigen::matrix_ref<scalar>& Y, const eigen::matrix_ref<scalar>& T) { return Negative_log_likelihood_loss_rowwise_gradient(Y, T); });

  /////////////////////////////////////////////////////////////////////////
  //                       softmax functions
  /////////////////////////////////////////////////////////////////////////

  m.def("softmax_colwise", [](const eigen::matrix_ref<scalar>& X) { return softmax_colwise(X); });
  m.def("softmax_colwise_jacobian", [](const eigen::matrix_ref<scalar>& X) { return softmax_colwise_jacobian(X); });
  m.def("stable_softmax_colwise", [](const eigen::matrix_ref<scalar>& X) { return stable_softmax_colwise(X); });
  m.def("log_softmax_colwise", [](const eigen::matrix_ref<scalar>& X) { return log_softmax_colwise(X); });
  m.def("stable_log_softmax_colwise", [](const eigen::matrix_ref<scalar>& X) { return stable_log_softmax_colwise(X); });
  m.def("log_softmax_colwise_jacobian", [](const eigen::matrix_ref<scalar>& X) { return log_softmax_colwise_jacobian(X); });
  m.def("softmax_rowwise", [](const eigen::matrix_ref<scalar>& X) { return softmax_rowwise(X); });
  m.def("softmax_rowwise_jacobian", [](const eigen::matrix_ref<scalar>& X) { return softmax_rowwise_jacobian(X); });
  m.def("stable_softmax_rowwise", [](const eigen::matrix_ref<scalar>& X) { return stable_softmax_rowwise(X); });
  m.def("log_softmax_rowwise", [](const eigen::matrix_ref<scalar>& X) { return log_softmax_rowwise(X); });
  m.def("log_softmax_rowwise_jacobian", [](const eigen::matrix_ref<scalar>& X) { return log_softmax_rowwise_jacobian(X); });
  m.def("stable_log_softmax_rowwise", [](const eigen::matrix_ref<scalar>& X) { return stable_log_softmax_rowwise(X); });

  /////////////////////////////////////////////////////////////////////////
  //                       random
  /////////////////////////////////////////////////////////////////////////

  m.def("manual_seed", manual_seed);

  /////////////////////////////////////////////////////////////////////////
  //                       global timer
  /////////////////////////////////////////////////////////////////////////

  m.def("nerva_timer_enable", nerva_timer_enable);
  m.def("nerva_timer_disable", nerva_timer_disable);
  m.def("nerva_timer_suspend", nerva_timer_suspend);
  m.def("nerva_timer_resume", nerva_timer_resume);
  m.def("nerva_timer_start", nerva_timer_start);
  m.def("nerva_timer_stop", nerva_timer_stop);

  /////////////////////////////////////////////////////////////////////////
  //                       version
  /////////////////////////////////////////////////////////////////////////

  m.attr("__version__") = "0.20";
}
