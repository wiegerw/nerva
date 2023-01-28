// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/masking.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MASKING_H
#define NERVA_NEURAL_NETWORKS_MASKING_H

#include <Eigen/Dense>
#include <stdexcept>

namespace nerva {

// Creates a boolean mask of A such that non-zero values correspond to true.
template <typename Scalar, int MatrixLayout>
Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> create_binary_mask(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask(A.rows(), A.cols());

  for (int i = 0; i < A.rows(); i++) 
  {
    for (int j = 0; j < A.cols(); j++) 
    {
      mask(i, j) = A(i, j) != 0;
    }
  }
  return mask;
}

// Applies a binary mask to A. Entries corresponding to false are set to zero.
template <typename Scalar, int MatrixLayout>
void apply_binary_mask(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A,
                       const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& mask
                      )
{
  if (A.rows() != mask.rows() || A.cols() != mask.cols())
  {
    throw std::runtime_error("Could not apply binary mask, because the dimensions do not match.");
  }
  A = A.cwiseProduct(mask.cast<Scalar>());
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_MASKING_H
