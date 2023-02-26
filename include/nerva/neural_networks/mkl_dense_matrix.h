// Copyright: Wieger Wesselink 2022-present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_dense_matrix.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H
#define NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H

#include <mkl.h>
#include <stdexcept>
#include <vector>

namespace nerva::mkl {

enum class matrix_layout
{
  row_major,
  column_major
};

// This class can be used to wrap an Eigen matrix (or NumPy etc.)
template <typename Scalar, matrix_layout layout>
class dense_matrix_view;

template <typename Scalar, matrix_layout layout>
class dense_matrix_view<const Scalar, layout>
{
  protected:
    long m; // number of rows
    long n; // number of columns
    const Scalar* values;

  public:
    dense_matrix_view(long m_, long n_, const Scalar* values_)
      : m(m_), n(n_), values(values_)
    {}

    [[nodiscard]] long rows() const
    {
      return m;
    }

    [[nodiscard]] long cols() const
    {
      return n;
    }

    const Scalar* data() const
    {
      return values;
    }
};

template <typename Scalar, matrix_layout layout>
class dense_matrix_view
{
  protected:
    long m; // number of rows
    long n; // number of columns
    Scalar* values;

  public:
    dense_matrix_view(long m_, long n_, Scalar* values_)
      : m(m_), n(n_), values(values_)
    {}

    [[nodiscard]] long rows() const
    {
      return m;
    }

    [[nodiscard]] long cols() const
    {
      return n;
    }

    Scalar* data()
    {
      return values;
    }

    const Scalar* data() const
    {
      return values;
    }
};

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_DENSE_MATRIX_H
