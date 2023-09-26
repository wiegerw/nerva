// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_dense_vector.h
/// \brief add your file description here.

#pragma once

#include <mkl.h>
#include <type_traits>

namespace nerva {

template <typename Scalar_>
class dense_vector_view
{
  public:
    using Scalar = Scalar_;

  protected:
    Scalar* m_data;
    long m_size;
    long m_increment;

  public:
    dense_vector_view(Scalar* data, long size, long increment = 1)
      : m_data(data), m_size(size), m_increment(increment)
    { }

    [[nodiscard]] long size() const
    {
      return m_size;
    }

    [[nodiscard]] long increment() const
    {
      return m_increment;
    }

    Scalar* data()
    {
      return m_data;
    }

    const Scalar* data() const
    {
      return m_data;
    }

    Scalar operator()(long i) const
    {
      return m_data[i * m_increment];
    }

    Scalar& operator()(long i, long j)
    {
      return m_data[i * m_increment];
    }

    // N.B. Not very efficient
    bool operator==(const dense_vector_view<Scalar>& other) const
    {
      if (m_size != other.m_size)
      {
        return false;
      }
      for (auto i = 0; i < m_size; i++)
      {
        if ((*this)(i) != other(i))
        {
          return false;
        }
      }
      return true;
    }
};

// y := a*x + y
template <typename Scalar>
void cblas_axpy(Scalar a, const dense_vector_view<Scalar>& x, dense_vector_view<Scalar>& y)
{
  assert(x.size() == y.size());
  if constexpr (std::is_same_v<Scalar, float>)
  {
    cblas_saxpy(x.size(), a, x.data(), x.increment(), y.data(), y.increment());
  }
  else
  {
    cblas_daxpy(x.size(), a, x.data(), x.increment(), y.data(), y.increment());
  }
};

// returns elements_sum(abs(x))
template <typename Scalar>
Scalar cblas_asum(dense_vector_view<Scalar>& x)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return cblas_sasum(x.size(), x.data(), x.increment());
  }
  else
  {
    return cblas_dasum(x.size(), x.data(), x.increment());
  }
};

// y := x
template <typename Scalar>
void cblas_copy(const dense_vector_view<Scalar>& x, dense_vector_view<Scalar>& y)
{
  assert(x.size() == y.size());
  if constexpr (std::is_same_v<Scalar, float>)
  {
    cblas_scopy(x.size(), x.data(), x.increment(), y.data(), y.increment());
  }
  else
  {
    cblas_dcopy(x.size(), x.data(), x.increment(), y.data(), y.increment());
  }
};

// returns elements_sum(hadamard(x, y))
template <typename Scalar>
Scalar cblas_dot(const dense_vector_view<Scalar>& x, dense_vector_view<Scalar>& y)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return cblas_sdot(x.size(), x.data(), x.increment(), y.data(), y.increment());
  }
  else
  {
    return cblas_ddot(x.size(), x.data(), x.increment(), y.data(), y.increment());
  }
};

// returns elements_sum(abs(sqrt(x)))
template <typename Scalar>
Scalar cblas_nrm2(dense_vector_view<Scalar>& x)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return cblas_snrm2(x.size(), x.data(), x.increment());
  }
  else
  {
    return cblas_dnrm2(x.size(), x.data(), x.increment());
  }
};

// x := a*x
template <typename Scalar>
void cblas_scal(Scalar a, dense_vector_view<Scalar>& x)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    cblas_sscal(x.size(), a, x.data(), x.increment());
  }
  else
  {
    cblas_dscal(x.size(), a, x.data(), x.increment());
  }
};

// x, y := y, x
template <typename Scalar>
void cblas_swap(const dense_vector_view<Scalar>& x, dense_vector_view<Scalar>& y)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return cblas_sswap(x.size(), x.data(), x.increment(), y.data(), y.increment());
  }
  else
  {
    return cblas_dswap(x.size(), x.data(), x.increment(), y.data(), y.increment());
  }
};

// Returns the index of the element with maximum absolute value.
template <typename Scalar>
CBLAS_INDEX cblas_iamax(dense_vector_view<Scalar>& x)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return cblas_isamax(x.size(), x.data(), x.increment());
  }
  else
  {
    return cblas_idamax(x.size(), x.data(), x.increment());
  }
};

// Returns the index of the element with minimum absolute value.
template <typename Scalar>
CBLAS_INDEX cblas_iamin(dense_vector_view<Scalar>& x)
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return cblas_isamin(x.size(), x.data(), x.increment());
  }
  else
  {
    return cblas_idamin(x.size(), x.data(), x.increment());
  }
};

} // namespace nerva
