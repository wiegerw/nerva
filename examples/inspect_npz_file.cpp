#include "nerva/neural_networks/numpy_eigen.h"
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>

namespace py = pybind11;
using namespace nerva;

void print_dict(const py::dict& data)
{
  for (const auto& item: data)
  {
    std::string key = item.first.cast<std::string>();
    if (key[0] == 'W')
    {
      eigen::print_numpy_matrix(key, eigen::load_float_matrix_from_dict(data, key).transpose());
    }
    else if (key[0] == 'b')
    {
      eigen::print_numpy_matrix(key, eigen::load_float_vector_from_dict(data, key).transpose());
    }
  }
}

void load_npz(const std::string& filename)
{
  auto np = py::module::import("numpy");
  py::dict d = np.attr("load")(filename);
  print_dict(d);
}

int main(int argc, char *argv[])
{
  py::scoped_interpreter guard{};  // Initialize the interpreter
  std::string filename(argv[1]);
  load_npz(filename);

  return 0;
}

