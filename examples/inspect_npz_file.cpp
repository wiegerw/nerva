#include "nerva/datasets/dataset.h"
#include "nerva/neural_networks/numpy_eigen.h"
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>

namespace py = pybind11;

void load_npz(const std::string& filename)
{
  auto np = py::module::import("numpy");
  py::dict d = np.attr("load")(filename);

  auto Xtrain = nerva::eigen::from_numpy(d["Xtrain"].cast<py::array_t<float>>());
  auto Ttrain = nerva::eigen::from_numpy_1d(d["Ttrain"].cast<py::array_t<long>>());
  auto Xtest  = nerva::eigen::from_numpy(d["Xtest"].cast<py::array_t<float>>());
  auto Ttest  = nerva::eigen::from_numpy_1d(d["Ttest"].cast<py::array_t<long>>());

  nerva::eigen::print_numpy_matrix("Xtrain", Xtrain);
  nerva::eigen::print_numpy_matrix("Ttrain", Ttrain.transpose());
  nerva::eigen::print_numpy_matrix("Xtest", Xtest);
  nerva::eigen::print_numpy_matrix("Ttest", Ttest.transpose());

  nerva::datasets::dataset data(Xtrain, Ttrain, Xtest, Ttest);
  data.info();
}

int main(int argc, char *argv[])
{
  py::scoped_interpreter guard{};  // Initialize the interpreter
  std::string filename(argv[1]);
  load_npz(filename);

  return 0;
}

