#include "nerva/neural_networks/numpy_eigen.h"
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>

namespace py = pybind11;
using namespace nerva;

void load_npz(const std::string& filename)
{
  auto np = py::module::import("numpy");
  py::dict data = np.attr("load")(filename);
  eigen::print_dict(data);
}

int main(int argc, char *argv[])
{
  py::scoped_interpreter guard{};  // Initialize the interpreter
  std::string filename(argv[1]);
  load_npz(filename);

  return 0;
}

