// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/examples/io.cpp
/// \brief add your file description here.

#include "nerva/utilities/command_line_tool.h"
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;
using namespace nerva;

py::dict create_dict()
{
  auto np = py::module::import("numpy");
  auto io = py::module::import("io");

  Eigen::MatrixXf A {
    {1.1, 2.2, 3.3},
    {4.4, 5.5, 6.6}
  };
  py::array_t<float> A_data({A.rows(), A.cols()});
  Eigen::Map<Eigen::MatrixXf>(A_data.mutable_data(), A.rows(), A.cols()) = A;

  std::string B = "Hello, world!";

  Eigen::MatrixXi C {
    {3, 8},
    {1, 7},
    {9, 5}
  };
  py::array_t<int> C_data({C.rows(), C.cols()});
  Eigen::Map<Eigen::MatrixXi>(C_data.mutable_data(), C.rows(), C.cols()) = C;

  py::dict data;
  data["A"] = A_data;
  data["B"] = B;
  data["C"] = C_data;

  return data;
}

void print_array(const py::array& A)
{
  assert(A.ndim() == 2);
  auto shape = A.shape();

  if (A.dtype().is(py::dtype::of<int>()))
  {
    std::cout << Eigen::Map<Eigen::MatrixXi>(const_cast<int*>(A.cast<py::array_t<int>>().data()), shape[0], shape[1]) << std::endl;
  }
  else if (A.dtype().is(py::dtype::of<float>()))
  {
    std::cout << Eigen::Map<Eigen::MatrixXf>(const_cast<float*>(A.cast<py::array_t<float>>().data()), shape[0], shape[1]) << std::endl;
  }
}

void print_dict(const py::dict& d)
{
  for (const auto& [key, value] : d)
  {
    py::print(key);
    py::print(" -> ");
    if (py::isinstance<py::str>(value))
    {
      py::print(value.cast<std::string>());
    }
    else if (py::isinstance<py::array>(value))
    {
      auto array = value.cast<pybind11::array>();
      print_array(array);
    }
    py::print();
  }
}

py::dict load_dict(const std::string& filename, const std::string& format, bool allow_pickle)
{
  std::cout << "Loading dict from " << filename << std::endl;
  py::dict result;
  if (format == "npz")
  {
    auto np = py::module::import("numpy");
    if (allow_pickle)
    {
      result = np.attr("load")(filename, py::arg("allow_pickle") = allow_pickle);
    }
    else
    {
      result = np.attr("load")(filename);
    }
  }
  else
  {
    throw std::runtime_error("cannot load dict in format " + format);
  }
  return result;
}

void save_dict(const std::string& filename, const std::string& format, const py::dict& d, bool allow_pickle)
{
  std::cout << "Saving dict to " << filename << std::endl;
  if (format == "npz")
  {
    auto np = py::module::import("numpy");
    if (allow_pickle)
    {
      np.attr("savez")(filename, d, py::arg("allow_pickle") = allow_pickle);
    }
    else
    {
      np.attr("savez")(filename, d);
    }
  }
  else
  {
    throw std::runtime_error("cannot save dict in format " + format);
  }
}

class tool: public command_line_tool
{
  protected:
    std::string filename;
    std::string format = "npz";
    bool save = false;
    bool load = false;
    bool print = false;
    bool allow_pickle = false;

    void add_options(lyra::cli& cli) override
    {
      cli |= lyra::opt(load)["--load"]("Load the dictionary");
      cli |= lyra::opt(save)["--save"]("Save the dictionary");
      cli |= lyra::opt(print)["--print"]("Print the dictionary");
      cli |= lyra::opt(allow_pickle)["--pickle"]("Allow pickle");
      cli |= lyra::opt(format, "value")["--format"]("The file format (npy, npz or h5py)");
      cli |= lyra::arg(filename, "value").required()("The input or output file");
    }

    void info() const
    {
      std::cout << std::boolalpha;
      std::cout << "filename     = " << filename << std::endl;
      std::cout << "format       = " << format << std::endl;
      std::cout << "save         = " << save << std::endl;
      std::cout << "load         = " << load << std::endl;
      std::cout << "print        = " << print << std::endl;
      std::cout << "allow_pickle = " << allow_pickle << std::endl;
    }

    std::string description() const override
    {
      return "Load/save test of dictionaries";
    }

    bool run() override
    {
      info();
      py::dict data;

      if (load)
      {
        data = load_dict(filename, format, allow_pickle);
      }
      else
      {
        data = create_dict();
      }

      if (print)
      {
        print_dict(data);
      }

      if (save)
      {
        save_dict(filename, format, data, allow_pickle);
      }

      return true;
    }
};

int main(int argc, const char** argv)
{
  pybind11::scoped_interpreter guard{};
  return tool().execute(argc, argv);
}
