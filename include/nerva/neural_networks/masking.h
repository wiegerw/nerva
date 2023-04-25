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

#include "nerva/neural_networks/multilayer_perceptron.h"
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

class mlp_masking
{
    using boolean_matrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

  protected:
    std::vector<boolean_matrix> masks;

  public:
    explicit mlp_masking(const multilayer_perceptron& M)
    {
      // create a binary mask for every sparse linear layer in M
      for (auto& layer: M.layers)
      {
        if (auto slayer = dynamic_cast<sparse_linear_layer*>(layer.get()))
        {
          masks.push_back(create_binary_mask(mkl::to_eigen(slayer->W)));
        }
      }
    }

    // applies the masks to the dense linear layers in M
    void apply(multilayer_perceptron& M)
    {
      int unsigned index = 0;
      for (auto& layer: M.layers)
      {
        if (auto dlayer = dynamic_cast<dense_linear_layer*>(layer.get()))
        {
          apply_binary_mask(dlayer->W, masks[index++]);
          if (index >= masks.size())
          {
            break;
          }
        }
      }
    }
};

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_MASKING_H
