// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file src/python-bindings.cpp
/// \brief add your file description here.

#include "nerva/neural_networks/batch_normalization_layer.h"
#include "nerva/datasets/dataset.h"
#include "nerva/neural_networks/dropout_layers.h"
#include "nerva/neural_networks/learning_rate_schedulers.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/random.h"
#include "nerva/neural_networks/regrowth.h"
#include "nerva/neural_networks/training.h"
#include "nerva/neural_networks/weights.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <sstream>

namespace py = pybind11;
using namespace nerva;

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void export_matrix_to_numpy(const std::string& filename, const eigen::matrix& A)
{
  namespace py = pybind11;
  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "wb");

  np.attr("save")(file, eigen::to_numpy(A));
  file.attr("close")();
  print_numpy_matrix("export to " + filename, A);
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void export_default_matrices_to_numpy(const std::string& filename1, const std::string& filename2)
{
  eigen::matrix A {
    {1, 2, 3},
    {4, 5, 6}
  };
  export_matrix_to_numpy(filename1, A);
  export_matrix_to_numpy(filename2, A.transpose());
}

// Precondition: the python interpreter must be running.
// This can be enforced using `py::scoped_interpreter guard{};`
inline
void import_matrix_from_numpy(const std::string& filename)
{
  namespace py = pybind11;
  auto np = py::module::import("numpy");
  auto io = py::module::import("io");
  auto file = io.attr("open")(filename, "rb");

  eigen::matrix A = eigen::from_numpy(np.attr("load")(file).cast<py::array_t<float>>());
  file.attr("close")();
  print_numpy_matrix("import from " + filename, A);
}

PYBIND11_MODULE(nervalib, m)
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
    .def("import_cifar10_from_npz", &datasets::dataset::import_cifar10_from_npz)
    ;

  py::class_<datasets::dataset_view, std::shared_ptr<datasets::dataset_view>>(m, "DataSetView")
    .def(py::init<datasets::matrix_ref, datasets::matrix_ref, datasets::matrix_ref, datasets::matrix_ref>(), py::return_value_policy::copy)
    .def("info", &datasets::dataset_view::info)
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
    ;

  py::class_<time_based_scheduler, learning_rate_scheduler, std::shared_ptr<time_based_scheduler>>(m, "time_based_scheduler")
    .def(py::init<scalar, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<step_based_scheduler, learning_rate_scheduler, std::shared_ptr<step_based_scheduler>>(m, "step_based_scheduler")
    .def(py::init<scalar, scalar, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<multi_step_lr_scheduler, learning_rate_scheduler, std::shared_ptr<multi_step_lr_scheduler>>(m, "multi_step_lr_scheduler")
    .def(py::init<scalar, std::vector<unsigned int>, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<exponential_scheduler, learning_rate_scheduler, std::shared_ptr<exponential_scheduler>>(m, "exponential_scheduler")
    .def(py::init<scalar, scalar>(), py::return_value_policy::copy)
    ;

  /////////////////////////////////////////////////////////////////////////
  //                       optimizers
  /////////////////////////////////////////////////////////////////////////

  py::class_<layer_optimizer, std::shared_ptr<layer_optimizer>>(m, "layer_optimizer")
    .def("update", &layer_optimizer::update)
    ;

  py::class_<gradient_descent_optimizer<eigen::matrix>, layer_optimizer, std::shared_ptr<gradient_descent_optimizer<eigen::matrix>>>(m, "gradient_descent_optimizer")
    .def(py::init<eigen::matrix&, eigen::matrix&, eigen::vector&, eigen::vector&>(), py::return_value_policy::copy)
    ;

  py::class_<momentum_optimizer<eigen::matrix>, layer_optimizer, std::shared_ptr<momentum_optimizer<eigen::matrix>>>(m, "momentum_optimizer")
    .def(py::init<eigen::matrix&, eigen::matrix&, eigen::vector&, eigen::vector&, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<nesterov_optimizer<eigen::matrix>, layer_optimizer, std::shared_ptr<nesterov_optimizer<eigen::matrix>>>(m, "nesterov_optimizer")
    .def(py::init<eigen::matrix&, eigen::matrix&, eigen::vector&, eigen::vector&, scalar>(), py::return_value_policy::copy)
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
    .def("set_optimizer", [](dense_linear_layer& layer, const std::string& text) { set_optimizer(layer, text); })
    ;

  py::class_<dense_relu_layer, dense_linear_layer, std::shared_ptr<dense_relu_layer>>(m, "relu_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_all_relu_layer, dense_linear_layer, std::shared_ptr<dense_all_relu_layer>>(m, "all_relu_layer")
    .def(py::init<scalar, std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
    ;

  py::class_<dense_leaky_relu_layer, dense_linear_layer, std::shared_ptr<dense_leaky_relu_layer>>(m, "leaky_relu_layer")
    .def(py::init<scalar, std::size_t, std::size_t, std::size_t>(), py::return_value_policy::copy)
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
    .def("set_optimizer", [](dense_linear_dropout_layer& layer, const std::string& text) { set_optimizer(layer, text); })
    ;

  py::class_<dense_relu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_relu_dropout_layer>>(m, "relu_dropout_layer")
    .def(py::init<std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_all_relu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_all_relu_dropout_layer>>(m, "all_relu_dropout_layer")
    .def(py::init<scalar, std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
    ;

  py::class_<dense_leaky_relu_dropout_layer, dense_linear_layer, std::shared_ptr<dense_leaky_relu_dropout_layer>>(m, "leaky_relu_dropout_layer")
    .def(py::init<scalar, std::size_t, std::size_t, std::size_t, scalar>(), py::return_value_policy::copy)
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

  //--- sparse layers ---//
  py::class_<sparse_linear_layer, neural_network_layer, std::shared_ptr<linear_layer<mkl::sparse_matrix_csr<scalar>>>>(m, "sparse_linear_layer")
    .def(py::init([](std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_linear_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
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
    .def("set_optimizer", [](sparse_linear_layer& layer, const std::string& text) { set_optimizer(layer, text); })
    .def("regrow", [](sparse_linear_layer& layer, scalar zeta, bool separate_positive_negative, weight_initialization w)
        {
          regrow(layer.W, zeta, w, separate_positive_negative, nerva_rng);
        })
    ;

  py::class_<sparse_hyperbolic_tangent_layer, sparse_linear_layer, std::shared_ptr<hyperbolic_tangent_layer<mkl::sparse_matrix_csr<scalar>>>>(m, "sparse_hyperbolic_tangent_layer")
    .def(py::init([](std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_hyperbolic_tangent_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
    ;

  py::class_<sparse_relu_layer, sparse_linear_layer, std::shared_ptr<relu_layer<mkl::sparse_matrix_csr<scalar>>>>(m, "sparse_relu_layer")
    .def(py::init([](std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_relu_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
    ;

  py::class_<sparse_leaky_relu_layer, sparse_linear_layer, std::shared_ptr<leaky_relu_layer<mkl::sparse_matrix_csr<scalar>>>>(m, "sparse_leaky_relu_layer")
    .def(py::init([](scalar alpha, std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_leaky_relu_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
    ;

  py::class_<sparse_all_relu_layer, sparse_linear_layer, std::shared_ptr<all_relu_layer<mkl::sparse_matrix_csr<scalar>>>>(m, "sparse_all_relu_layer")
    .def(py::init([](scalar alpha, std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_all_relu_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
    ;

  py::class_<sparse_sigmoid_layer, sparse_linear_layer, std::shared_ptr<sparse_sigmoid_layer>>(m, "sparse_sigmoid_layer")
    .def(py::init([](std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_sigmoid_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
    ;

  py::class_<sparse_softmax_layer, sparse_linear_layer, std::shared_ptr<sparse_softmax_layer>>(m, "sparse_softmax_layer")
    .def(py::init([](std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_softmax_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
    ;

  py::class_<sparse_log_softmax_layer, sparse_linear_layer, std::shared_ptr<sparse_log_softmax_layer>>(m, "sparse_log_softmax_layer")
    .def(py::init([](std::size_t D, std::size_t K, std::size_t batch_size, scalar sparsity)
                  {
                    auto layer = std::make_shared<sparse_log_softmax_layer>(D, K, batch_size);
                    initialize_sparse_weights<scalar>(*layer, sparsity, nerva_rng);
                    return layer;
                  }))
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

  /////////////////////////////////////////////////////////////////////////
  //                       multilayer perceptron
  /////////////////////////////////////////////////////////////////////////

  py::class_<multilayer_perceptron, std::shared_ptr<multilayer_perceptron>>(m, "MLP")
    .def(py::init<>(), py::return_value_policy::copy)
    .def("__str__", [](const multilayer_perceptron& M) { return M.to_string(); })
    .def_readwrite("layers", &multilayer_perceptron::layers)
    .def("feedforward", [](multilayer_perceptron& M, const eigen::matrix& X) { eigen::matrix Y; M.feedforward(X, Y); return Y; })
    .def("backpropagate", &multilayer_perceptron::backpropagate)
    .def("renew_dropout_mask", &multilayer_perceptron::renew_dropout_mask)
    .def("optimize", &multilayer_perceptron::optimize)
    .def("regrow", &multilayer_perceptron::regrow)
    .def("append_layer", [](multilayer_perceptron& M, const std::shared_ptr<neural_network_layer>& layer) { M.layers.push_back(layer); })
    .def("info", &multilayer_perceptron::info)
    .def("export_weights_npy", [](const multilayer_perceptron& M, const std::string& filename) { export_weights_to_npy(M, filename); })
    .def("import_weights_npy", [](multilayer_perceptron& M, const std::string& filename) { import_weights_from_npy(M, filename); })
    .def("export_bias_npy", [](const multilayer_perceptron& M, const std::string& filename) { export_bias_to_npy(M, filename); })
    .def("import_bias_npy", [](multilayer_perceptron& M, const std::string& filename) { import_bias_from_npy(M, filename); })
    .def("export_weights_npz", [](const multilayer_perceptron& M, const std::string& filename) { export_weights_to_npz(M, filename); })
    .def("import_weights_npz", [](multilayer_perceptron& M, const std::string& filename) { import_weights_from_npz(M, filename); })
    ;

  py::class_<mlp_masking, std::shared_ptr<mlp_masking>>(m, "MLPMasking")
    .def(py::init<const multilayer_perceptron&>(), py::return_value_policy::copy)
    .def("apply", &mlp_masking::apply)
    ;

  m.def("save_model_weights_to_npy", save_model_weights_to_npy);

  /////////////////////////////////////////////////////////////////////////
  //                       weights
  /////////////////////////////////////////////////////////////////////////

  py::enum_<weight_initialization>(m, "Weights")
    .value("Default", weight_initialization::default_, "Default")
    .value("He", weight_initialization::he, "He")
    .value("Xavier", weight_initialization::xavier, "Xavier")
    .value("XavierNormalized", weight_initialization::xavier_normalized, "XavierNormalized")
    .value("Uniform", weight_initialization::uniform, "Uniform")
    .value("Zero", weight_initialization::zero, "Zero")
    .value("Ten", weight_initialization::ten, "Ten")  // used for testing
    ;

  m.def("initialize_weights", initialize_weights<eigen::matrix>);
  m.def("regrow", [](eigen::matrix_ref<scalar> W, scalar zeta, weight_initialization w)
        {
          auto f = create_weight_initializer(W, w, nerva_rng);
          long nonzero_count = (W.array() != 0).count();
          long k = std::lround(zeta * static_cast<scalar>(nonzero_count));
          regrow_threshold(W, k, f, nerva_rng);
        });

  /////////////////////////////////////////////////////////////////////////
  //                       training
  /////////////////////////////////////////////////////////////////////////

  py::class_<sgd_options>(m, "SGDOptions")
    .def(py::init<>(), py::return_value_policy::copy)
    .def_readwrite("batch_size", &sgd_options::batch_size)
    .def_readwrite("epochs", &sgd_options::epochs)
    .def_readwrite("debug", &sgd_options::debug)
    .def_readwrite("shuffle", &sgd_options::shuffle)
    .def_readwrite("statistics", &sgd_options::statistics)
    .def("info", &sgd_options::info)
    ;

  m.def("minibatch_gradient_descent", minibatch_gradient_descent<multilayer_perceptron, datasets::dataset_view, std::mt19937>);
  m.def("compute_loss", compute_loss_batch<multilayer_perceptron>);
  m.def("compute_accuracy", compute_accuracy_batch<multilayer_perceptron, datasets::matrix_ref>);
  m.def("compute_statistics", compute_statistics_batch<multilayer_perceptron, datasets::dataset_view>);
  m.def("set_num_threads", mkl_set_num_threads);

  /////////////////////////////////////////////////////////////////////////
  //                       activation functions
  /////////////////////////////////////////////////////////////////////////

  m.def("relu", [](const eigen::matrix_ref<scalar>& X) { return relu_activation()(X); });
  m.def("sigmoid", [](const eigen::matrix_ref<scalar>& X) { return sigmoid_activation()(X); });
  m.def("softmax", [](const eigen::matrix_ref<scalar>& X) { return softmax_activation()(X); });
  m.def("log_softmax", [](const eigen::matrix_ref<scalar>& X) { return log_softmax_activation()(X); });
  m.def("hyperbolic_tangent", [](const eigen::matrix_ref<scalar>& X) { return hyperbolic_tangent_activation()(X); });

  /////////////////////////////////////////////////////////////////////////
  //                       random
  /////////////////////////////////////////////////////////////////////////

  m.def("manual_seed", manual_seed);

  /////////////////////////////////////////////////////////////////////////
  //                       testing
  /////////////////////////////////////////////////////////////////////////

  m.def("export_matrix_to_numpy", export_matrix_to_numpy);
  m.def("import_matrix_from_numpy", import_matrix_from_numpy);
  m.def("export_default_matrices_to_numpy", export_default_matrices_to_numpy);

  m.attr("__version__") = "0.12";
}
