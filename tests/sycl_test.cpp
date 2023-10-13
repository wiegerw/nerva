// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mask_test.cpp
/// \brief Tests for masking.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "nerva/neural_networks/mkl_dense_vector.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "doctest/doctest.h"
#include <iostream>

#ifdef NERVA_SYCL
#include <sycl/sycl.hpp>

TEST_CASE("test_khronos_example")
{
  using namespace sycl;
  size_t data[1024];

  {
    queue myQueue;
    buffer<size_t, 1> resultBuf{ data, range<1>{1024}};
    myQueue.submit([&](handler& cgh)
    {
      auto writeResult = resultBuf.get_access<access::mode::write>(cgh);
      cgh.parallel_for(range<1>{1024}, [=](id<1> idx)
      {
        writeResult[idx] = idx[0];
      });
    }).wait();
  }

  for (int i = 0; i < 1024; i++)
  {
    std::cout << "data[" << i << "] = " << data[i] << std::endl;
  }
}

TEST_CASE("test_sycl1")
{
  std::vector<float> x = { 1, 2, 3, 4 };
  std::vector<float> y = { 3, 4, 3, 4 };
  std::vector<float> z = { 0, 0, 0, 0 };
  std::vector<float> expected = { 11, 16, 15, 20 };

  float a = 2.0;
  float b = 3.0;

  {
    sycl::queue q;
    sycl::buffer<float, 1> x_buffer{ x.data(), sycl::range<1>{x.size()}};
    sycl::buffer<float, 1> y_buffer{ y.data(), sycl::range<1>{y.size()}};
    sycl::buffer<float, 1> z_buffer{ z.data(), sycl::range<1>{z.size()}};

    q.submit([&](sycl::handler& cgh)
    {
      auto x = x_buffer.get_access<sycl::access::mode::read>(cgh);
      auto y = y_buffer.get_access<sycl::access::mode::read>(cgh);
      auto z = z_buffer.get_access<sycl::access::mode::write>(cgh);
      std::size_t n = x.size();

      cgh.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i)
      {
        z[i] = a * x[i] + b * y[i];
      });
    }).wait();
  }

  for (int i = 0; i < x.size(); i++)
  {
    std::cout << "z[" << i << "] = " << z[i] << std::endl;
  }
  CHECK(z == expected);
}

TEST_CASE("test_sycl2")
{
  using namespace nerva;

  eigen::matrix x {{ 1, 2, 3, 4 }};
  eigen::matrix y {{ 3, 4, 3, 4 }};
  eigen::matrix z {{ 0, 0, 0, 0 }};
  eigen::matrix expected {{ 11, 16, 15, 20 }};

  scalar a = 2.0;
  scalar b = 3.0;

  auto x_view = mkl::make_dense_vector_view(x);
  auto y_view = mkl::make_dense_vector_view(y);
  auto z_view = mkl::make_dense_vector_view(z);

  assign_axby(a, b, x_view, y_view, z_view);

  CHECK(z == expected);
}

#endif // NERVA_SYCL
