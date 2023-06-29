// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file multilayer_perceptron_test.cpp
/// \brief Tests for multilayer perceptrons.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "nerva/neural_networks/multilayer_perceptron.h"
#include "nerva/neural_networks/loss_functions_colwise.h"
#include "nerva/utilities/stopwatch.h"
#include "nerva/utilities/string_utility.h"
#include <iostream>

using namespace nerva;

inline
void print_vector(std::ostream& out, const std::string& name, const eigen::vector& x)
{
  out << name << ' ';
  for (auto i = 0; i < x.size(); i++)
  {
    if (i != 0)
    {
      out << ' ';
    }
    out << std::setprecision(8) << x(i);
  }
  out << '\n';
}

bool compare_output(std::string output1, std::string output2)
{
  float epsilon = 0.000001;

  utilities::trim(output1);
  utilities::trim(output2);
  std::vector<std::string> lines1 = utilities::regex_split(output1, "\n");
  std::vector<std::string> lines2 = utilities::regex_split(output2, "\n");

  if (lines1.size() != lines2.size())
  {
    std::cout << "size difference " << lines1.size() << " " << lines2.size() << std::endl;
    return false;
  }

  for (unsigned int i = 0; i < lines1.size(); i++)
  {
    std::stringstream ss1(lines1[i]);
    std::stringstream ss2(lines2[i]);
    std::cout << lines1[i] << std::endl;
    std::cout << lines2[i] << std::endl;

    std::string name1;
    std::string name2;
    ss1 >> name1;
    ss2 >> name2;
    if (name1 != name2)
    {
      std::cout << "name difference " << name1 << " " << name2 << std::endl;
      return false;
    }

    while (true)
    {
      double value1;
      double value2;
      ss1 >> value1;
      ss2 >> value2;
      if (ss1.good() != ss2.good())
      {
        std::cout << "lines do not match" << std::endl;
        return false;
      }
      if (!ss1.good())
      {
        break;
      }
      if (std::abs(value1 - value2) > epsilon)
      {
        std::cout << "value difference " << value1 << " " << value2 << std::endl;
        return false;
      }
    }
  }
  return true;
}

void construct_mlp(multilayer_perceptron& M,
                   const eigen::matrix& w1,
                   const eigen::vector& b1,
                   const eigen::matrix& w2,
                   const eigen::vector& b2,
                   const eigen::matrix& w3,
                   const eigen::vector& b3,
                   std::size_t batch_size = 1
                  )
{
  std::size_t N1 = w1.cols();
  std::size_t N2 = w2.cols();
  std::size_t N3 = w3.cols();
  std::size_t N4 = w3.rows();

  auto layer1 = std::make_shared<relu_layer<eigen::matrix>>(N1, N2, batch_size);
  M.layers.push_back(layer1);
  layer1->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer1->W, layer1->DW, layer1->b, layer1->Db);
  layer1->W = w1;
  layer1->b = b1;

  auto layer2 = std::make_shared<relu_layer<eigen::matrix>>(N2, N3, batch_size);
  M.layers.push_back(layer2);
  layer2->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer2->W, layer2->DW, layer2->b, layer2->Db);
  layer2->W = w2;
  layer2->b = b2;

  auto layer3 = std::make_shared<linear_layer<eigen::matrix>>(N3, N4, batch_size);
  M.layers.push_back(layer3);
  layer3->optimizer = std::make_shared<gradient_descent_linear_layer_optimizer<eigen::matrix>>(layer3->W, layer3->DW, layer3->b, layer3->Db);
  layer3->W = w3;
  layer3->b = b3;
}

void test_case(const eigen::matrix& X,
               const eigen::matrix& T,
               const eigen::matrix& w1,
               const eigen::vector& b1,
               const eigen::matrix& w2,
               const eigen::vector& b2,
               const eigen::matrix& w3,
               const eigen::vector& b3,
               const std::string& expected_output
               )
{
  multilayer_perceptron M;
  std::size_t batch_size = 1;
  construct_mlp(M, w1, b1, w2, b2, w3, b3, batch_size);

  long N = X.cols();
  std::size_t N4 = w3.rows();
  eigen::matrix y(N4, 1);
  eigen::matrix dy(N4, 1);

  softmax_cross_entropy_loss loss;
  float eta = 0.001;

  std::ostringstream out;
  out << '\n';
  auto epochs = 2;
  for (auto epoch = 0; epoch < epochs; epoch++)
  {
    for (long i = 0; i < N; i++)
    {
      const auto& x = X.col(i);
      const auto& t = T.col(i);
      M.feedforward(x, y);
      dy = loss.gradient(y, t);
      print_vector(out, " y", y);
      print_vector(out, "dy", dy);
      M.backpropagate(y, dy);
      M.optimize(eta);
    }
  }
  CHECK(compare_output(expected_output, out.str()));
}

TEST_CASE("test_mlp1")
{
  eigen::matrix X
    {{0.5199999809265137, 0.9300000071525574, 0.15000000596046448, 0.7200000286102295}};

  eigen::matrix T
    {{1.0, 1.0, 1.0, 0.0},
     {0.0, 0.0, 0.0, 1.0}};

  eigen::matrix w1
    {{-0.8133288621902466},
     {0.3976917266845703}};

  eigen::vector b1 {{-0.6363444328308105, -0.2981985807418823}};

  eigen::matrix w2
    {{0.5963594317436218, 0.5629172325134277},
     {-0.09182517230510712, -0.6868335604667664}};

  eigen::vector b2 {{0.318002849817276, -0.5409443974494934}};

  eigen::matrix w3
    {{0.6574600338935852, -0.4956347942352295},
     {0.25072064995765686, -0.24014270305633545}};

  eigen::vector b3 {{0.48356249928474426, -0.353687584400177}};

  std::string expected_output{R"(
 y 0.69263667 -0.27395770
dy -0.27555990 0.27555984
 y 0.71953642 -0.26412356
dy -0.27216613 0.27216616
 y 0.69338977 -0.27450848
dy -0.27529967 0.27529961
 y 0.69376671 -0.27478361
dy 0.72483051 -0.72483045
 y 0.69277436 -0.27405933
dy -0.27551210 0.27551204
 y 0.71971917 -0.26421034
dy -0.27211273 0.27211279
 y 0.69352746 -0.27461004
dy -0.27525187 0.27525190
 y 0.69390428 -0.27488512
dy 0.72487813 -0.72487813
)"};
  test_case(X, T, w1, b1, w2, b2, w3, b3, expected_output);
}
