// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mkl.cpp
/// \brief add your file description here.

#include <iostream>
#include <random>
#include <lyra/lyra.hpp>
#include "fmt/format.h"
#include "omp.h"
// #include "mkl_omp_offload.h"
#include "nerva/utilities/stopwatch.h"
#include "nerva/utilities/command_line_tool.h"
#include "nerva/neural_networks/mkl_eigen.h"

using namespace nerva;

// A = B * C
void test_mult1(long m, long k, long n, const std::vector<float>& densities)
{
  std::cout << "--- test mult1 ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d}\n", m, n);
  std::cout << fmt::format("B = {:2d}x{:2d}\n", m, k);
  std::cout << fmt::format("C = {:2d}x{:2d}\n\n", k, n);

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density = {}\n", density);

    float a = -10;
    float b = 10;

    eigen::matrix A(m, n);
    eigen::fill_matrix_random(A, density, a, b, rng);

    eigen::matrix B(m, k);
    eigen::fill_matrix_random(B, 0, a, b, rng);

    eigen::matrix C(k, n);
    eigen::fill_matrix_random(C, 0, a, b, rng);

    mkl::sparse_matrix_csr<float> A1 = mkl::to_csr<float>(A);

    utilities::stopwatch watch;
    A = B * C;
    std::cout << fmt::format("A = B * C {:7.5f}s\n", watch.seconds());

    watch.reset();
    long batch_size = 100;
    mkl::sdd_product_batch(A1, B, C, batch_size);
    std::cout << fmt::format("A = B * C {:7.5f}s A sparse (batch size {:3d})\n", watch.seconds(), batch_size);

    watch.reset();
    batch_size = 10;
    mkl::sdd_product_batch(A1, B, C, batch_size);
    std::cout << fmt::format("A = B * C {:7.5f}s A sparse (batch size {:3d})\n", watch.seconds(), batch_size);

    watch.reset();
    batch_size = 5;
    mkl::sdd_product_batch(A1, B, C, batch_size);
    std::cout << fmt::format("A = B * C {:7.5f}s A sparse (batch size {:3d})\n", watch.seconds(), batch_size);

    if ((1 - density) * m <= 2000)
    {
      watch.reset();
      mkl::sdd_product_forloop_eigen(A1, B, C);
      std::cout << fmt::format("A = B * C {:7.5f}s A sparse (for-loop eigen dot product)\n", watch.seconds(), batch_size);

      watch.reset();
      mkl::sdd_product_forloop_mkl(A1, B, C);
      std::cout << fmt::format("A = B * C {:7.5f}s A sparse (for-loop mkl dot product)\n", watch.seconds(), batch_size);
    }

    std::cout << std::endl;
  }
}

// A = B * C
void test_mult2(long m, long k, long n, const std::vector<float>& densities)
{
  std::cout << "--- test mult2 ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d}\n", m, n);
  std::cout << fmt::format("B = {:2d}x{:2d}\n", m, k);
  std::cout << fmt::format("C = {:2d}x{:2d}\n\n", k, n);

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density = {}\n", density);

    float a = -10;
    float b = 10;

    eigen::matrix A(m, n);

    eigen::matrix B(m, k);
    eigen::fill_matrix_random(B, density, a, b, rng);

    eigen::matrix C(k, n);
    eigen::fill_matrix_random(C, float(0), a, b, rng);

    mkl::sparse_matrix_csr<float> B1 = mkl::to_csr<float>(B);

    utilities::stopwatch watch;
    A = B * C;
    std::cout << fmt::format("A = B * C {:7.5f}s\n", watch.seconds());

    watch.reset();
    mkl::dsd_product(A, B1, C);
    std::cout << fmt::format("A = B * C {:7.5f}s B sparse\n\n", watch.seconds());
  }
}

// A = B^T * C
void test_mult3(long m, long k, long n, const std::vector<float>& densities)
{
  std::cout << "--- test mult3 ---" << std::endl;
  std::cout << fmt::format("A = {:2d}x{:2d}\n", m, n);
  std::cout << fmt::format("B = {:2d}x{:2d}\n", k, m);
  std::cout << fmt::format("C = {:2d}x{:2d}\n\n", k, n);

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  for (float density: densities)
  {
    std::cout << fmt::format("density = {}\n", density);

    float a = -10;
    float b = 10;

    eigen::matrix A(m, n);

    eigen::matrix B(k, m);
    eigen::fill_matrix_random(B, density, a, b, rng);

    eigen::matrix C(k, n);
    eigen::fill_matrix_random(C, 0, a, b, rng);

    mkl::sparse_matrix_csr<float> B1 = mkl::to_csr<float>(B);

    utilities::stopwatch watch;
    A = B.transpose() * C;
    std::cout << fmt::format("A = B^T * C {:7.5f}s\n", watch.seconds());

    watch.reset();
    mkl::dsd_product(A, B1, C, float(0), float(1), SPARSE_OPERATION_TRANSPOSE);
    std::cout << fmt::format("A = B^T * C {:7.5f}s B sparse\n\n", watch.seconds());
  }
}

class tool: public command_line_tool
{
  protected:
    int m = 100;
    int k = 100;
    int n = 100;
    float alpha = 1.0;
    float beta = 0.0;
    std::string algorithm = "multiply";
    int threads = 1;
    // bool gpu = false;

    void add_options(lyra::cli& cli) override
    {
      cli |= lyra::opt(algorithm, "algorithm")["--algorithm"]["-a"]("The algorithm (mult1, add1, add2)");
      cli |= lyra::opt(m, "m")["--arows"]["-m"]("The number of rows of matrix A");
      cli |= lyra::opt(k, "k")["--acols"]["-k"]("The number of columns of matrix A");
      cli |= lyra::opt(n, "n")["--brows"]["-n"]("The number of rows of matrix B");
      cli |= lyra::opt(alpha, "alpha")["--alpha"]("The value alpha");
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

      // mkl_set_interface_layer(MKL_INTERFACE_LP64); // TODO: is this needed?

      std::vector<float> densities = {1.0, 0.5, 0.1, 0.05, 0.01, 0.001};
      if (algorithm == "mult1")
      {
        test_mult1(m, k, n, densities);
      }
      else if (algorithm == "mult2")
      {
        test_mult2(m, k, n, densities);
      }
      else if (algorithm == "mult3")
      {
        test_mult3(m, k, n, densities);
      }
      return true;
    }
};

int main(int argc, const char** argv)
{
  return tool().execute(argc, argv);
}
