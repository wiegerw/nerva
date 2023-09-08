// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mkl.cpp
/// \brief add your file description here.

#include <lyra/lyra.hpp>
#include "fmt/format.h"
#include "omp.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/utilities/stopwatch.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include <iostream>
#include <random>
#include <sstream>

using namespace nerva;

enum orientation
{
  colwise = 0,
  rowwise = 1
};

enum class layouts
{
  colwise_colwise,
  colwise_rowwise,
  rowwise_colwise,
  rowwise_rowwise
};

inline
std::string layout_string(int layout)
{
  return layout == colwise ? "colwise" : "rowwise";
}

inline
std::string layout_char(int layout)
{
  return layout == colwise ? "C" : "R";
}

template <typename Scalar>
std::string matrix_parameters(const std::string& name, const mkl::sparse_matrix_csr<Scalar>& A)
{
  return fmt::format("density({})={}", name, A.density());
}

template <typename Derived>
std::string matrix_parameters(const std::string& name, const Eigen::MatrixBase<Derived>& A)
{
  constexpr int MatrixLayout = Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;
  return fmt::format("{}={}", name, layout_string(MatrixLayout));
}

template <typename MatrixA, typename MatrixB, typename MatrixC>
std::string pp(const MatrixA& A, const MatrixB& B, const MatrixC& C)
{
  std::ostringstream out;
  out << matrix_parameters("A", A) << ", " << matrix_parameters("B", B) << ", " << matrix_parameters("C", C);
  return out.str();
}

// A = B * C
template <int MatrixLayoutB = colwise, int MatrixLayoutC = colwise>
void test_sdd_product(long m, long k, long n, const std::vector<float>& densities)
{
  std::cout << "--- testing A = B * C (sdd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} sparse\n", m, n);
  std::cout << fmt::format("B = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutB));
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density(A) = {}\n", density);

    float a = -10;
    float b = 10;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, colwise> A(m, n);
    eigen::fill_matrix_random(A, density, a, b, rng);
    mkl::sparse_matrix_csr<float> A1 = mkl::to_csr<float>(A);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutB> B(m, k);
    eigen::fill_matrix_random(B, 0, a, b, rng);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
    eigen::fill_matrix_random(C, 0, a, b, rng);

    // dense product
    utilities::stopwatch watch;
    A = B * C;
    auto seconds = watch.seconds();
    std::cout << fmt::format("{:8.5f}s ddd_product {}\n", seconds, pp(A, B, C));

    // sdd_product_batch
    for (long batch_size: {5, 10, 100})
    {
      watch.reset();
      mkl::sdd_product_batch(A1, B, C, batch_size);
      seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s sdd_product(batchsize={}, {})\n", seconds, batch_size, pp(A1, B, C));
    }

    if (density * m <= 2000)
    {
      watch.reset();
      mkl::sdd_product_forloop_eigen(A1, B, C);
      seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s sdd_product_forloop_eigen({})\n", seconds, pp(A1, B, C));

      watch.reset();
      mkl::sdd_product_forloop_mkl(A1, B, C);
      seconds = watch.seconds();
      std::cout << fmt::format("{:8.5f}s sdd_product_forloop_mkl()\n", seconds, pp(A1, B, C));
    }

    std::cout << std::endl;
  }
}

// A = B * C
template <int MatrixLayoutA = colwise, int MatrixLayoutC = colwise>
void test_dsd_product(long m, long k, long n, const std::vector<float>& densities)
{
  std::cout << "--- testing A = B * C (dsd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutA));
  std::cout << fmt::format("B = {:2d}x{:2d} sparse\n", m, n);
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density(B) = {}\n", density);

    float a = -10;
    float b = 10;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutA> A(m, n);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, colwise> B(m, k);
    eigen::fill_matrix_random(B, density, a, b, rng);
    mkl::sparse_matrix_csr<float> B1 = mkl::to_csr<float>(B);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
    eigen::fill_matrix_random(C, float(0), a, b, rng);

    utilities::stopwatch watch;
    A = B * C;
    auto seconds = watch.seconds();
    std::cout << fmt::format("{:8.5f}s ddd_product({})\n", seconds, pp(A, B, C));

    watch.reset();
    mkl::dsd_product(A, B1, C);
    seconds = watch.seconds();
    std::cout << fmt::format("{:8.5f}s dsd_product({})\n\n", seconds, pp(A, B1, C));
  }
}

// A = B^T * C
template <int MatrixLayoutA = colwise, int MatrixLayoutC = colwise>
void test_dsd_transpose_product(long m, long k, long n, const std::vector<float>& densities)
{
  std::cout << "--- testing A = B^T * C (dsd_product) ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d} dense  layout={}\n", m, k, layout_string(MatrixLayoutA));
  std::cout << fmt::format("B = {:2d}x{:2d} sparse\n", m, n);
  std::cout << fmt::format("C = {:2d}x{:2d} dense  layout={}\n\n", k, n, layout_string(MatrixLayoutC));

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density = {}\n", density);

    float a = -10;
    float b = 10;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutA> A(m, n);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, colwise> B(k, m);
    eigen::fill_matrix_random(B, density, a, b, rng);
    mkl::sparse_matrix_csr<float> B1 = mkl::to_csr<float>(B);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, MatrixLayoutC> C(k, n);
    eigen::fill_matrix_random(C, 0, a, b, rng);

    utilities::stopwatch watch;
    A = B.transpose() * C;
    auto seconds = watch.seconds();
    std::cout << fmt::format("{:8.5f}s ddd_product({})\n", seconds, pp(A, B, C));

    watch.reset();
    mkl::dsd_product(A, B1, C, float(0), float(1), SPARSE_OPERATION_TRANSPOSE);
    seconds = watch.seconds();
    std::cout << fmt::format("{:8.5f}s dsd_product({})\n\n", seconds, pp(A, B1, C));
  }
}

inline
layouts parse_layouts(const std::string& text)
{
  if (text == "cc")
  {
    return layouts::colwise_colwise;
  }
  else if (text == "cr")
  {
    return layouts::colwise_rowwise;
  }
  else if (text == "rc")
  {
    return layouts::rowwise_colwise;
  }
  else if (text == "rr")
  {
    return layouts::rowwise_rowwise;
  }
  throw std::runtime_error(fmt::format("unsupported layout {}", text));
}

class tool: public command_line_tool
{
  protected:
    int m = 100;
    int k = 100;
    int n = 100;
    float alpha = 1.0;
    float beta = 0.0;
    std::string algorithm;
    std::string layouts;
    int threads = 1;
    // bool gpu = false;

    void add_options(lyra::cli& cli) override
    {
      cli |= lyra::opt(algorithm, "algorithm")["--algorithm"]["-a"]("The algorithm (sdd, dsd, dsdt)");
      cli |= lyra::opt(m, "m")["--arows"]["-m"]("The number of rows of matrix A");
      cli |= lyra::opt(k, "k")["--acols"]["-k"]("The number of columns of matrix A");
      cli |= lyra::opt(n, "n")["--brows"]["-n"]("The number of rows of matrix B");
      cli |= lyra::opt(threads, "value")["--threads"]("The number of threads.");
      // cli |= lyra::opt(gpu)["--gpu"]("Use the GPU ");
    }

    std::string description() const override
    {
      return "Test the MKL library";
    }

    bool run() override
    {
      if (threads >= 1 && threads <= 8)
      {
        omp_set_num_threads(threads);
      }

      std::vector<float> densities = {1.0, 0.5, 0.1, 0.05, 0.01, 0.001};
      if (algorithm == "sdd")
      {
        test_sdd_product<colwise, colwise>(m, k, n, densities);
        test_sdd_product<colwise, rowwise>(m, k, n, densities);
        test_sdd_product<rowwise, colwise>(m, k, n, densities);
        test_sdd_product<rowwise, rowwise>(m, k, n, densities);
      }
      else if (algorithm == "dsd")
      {
        test_dsd_product<colwise, colwise>(m, k, n, densities);
//        test_dsd_product<colwise, rowwise>(m, k, n, densities);
//        test_dsd_product<rowwise, colwise>(m, k, n, densities);
        test_dsd_product<rowwise, rowwise>(m, k, n, densities);
      }
      else if (algorithm == "dsdt")
      {
        test_dsd_transpose_product<colwise, colwise>(m, k, n, densities);
//        test_dsd_transpose_product<colwise, rowwise>(m, k, n, densities);
//        test_dsd_transpose_product<rowwise, colwise>(m, k, n, densities);
        test_dsd_transpose_product<rowwise, rowwise>(m, k, n, densities);
      }
      return true;
    }
};

int main(int argc, const char** argv)
{
  return tool().execute(argc, argv);
}
