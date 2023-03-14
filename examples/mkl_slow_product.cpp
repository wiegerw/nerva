#include <Eigen/Dense>
#include "nerva/utilities/stopwatch.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include <iostream>
#include <numeric>
#include <random>

void product()
{
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(1024, 100);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(100, 1024);

  std::vector<double> mkl_times;
  std::vector<double> manual_times;

  // create a random permutation of 0, 1, ..., 1023
  std::vector<long> I(1024);
  std::iota(I.begin(), I.end(), 0);
  std::mt19937 rng{std::random_device{}()};
  std::shuffle(I.begin(), I.end(), rng);

  // The first MKL multiplication is always slow
  Eigen::MatrixXf C = A * B;

  nerva::utilities::stopwatch watch;
  for (std::size_t i = 0; i < I.size(); i++)
  {
    if (i % 10 == 0)
    {
      std::cout << "i = " << i << std::endl;
      watch.reset();
      Eigen::MatrixXf AB = A * B;
      auto seconds = watch.seconds();
      mkl_times.push_back(seconds);
      watch.reset();

      auto A1 = nerva::mkl::make_dense_matrix_view(A);
      auto B1 = nerva::mkl::make_dense_matrix_view(B);
      auto C1 = nerva::mkl::ddd_product_manual_loops<nerva::mkl::column_major>(A1, B1, false, false);
      seconds = watch.seconds();
      manual_times.push_back(seconds);
    }

    // fill the i-th row with small values
    for (long j = 0; j < 100; j++)
    {
      A(i, j) = 1e-40;
    }
  }

  std::cout << "--- mkl product times ---" << std::endl;
  for (double t: mkl_times)
  {
    std::cout << t << std::endl;
  }

  std::cout << "--- manual product times ---" << std::endl;
  for (double t: manual_times)
  {
    std::cout << t << std::endl;
  }
}

int main()
{
  product();

  return 0;
}
