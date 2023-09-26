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

//----------------------------------------------------------------------//
//                     BLAS level 1 routines
//----------------------------------------------------------------------//

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

//----------------------------------------------------------------------//
//                     VM Mathematical Functions
//----------------------------------------------------------------------//

template <typename Scalar, typename SingleOperation, typename DoubleOperation>
void vm_apply_unary_operation(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y, SingleOperation fsingle, DoubleOperation fdouble)
{
  assert(a.size() == y.size());
  const auto n = a.size();
  if constexpr (std::is_same_v<Scalar, float>)
  {
    fsingle(n, a.data(), a.increment(), y.data(), y.increment());
  }
  else
  {
    fdouble(n, a.data(), a.increment(), y.data(), y.increment());
  }
};

template <typename Scalar, typename SingleOperation, typename DoubleOperation>
void vm_apply_binary_operation(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y, SingleOperation fsingle, DoubleOperation fdouble)
{
  assert(a.size() == b.size() && a.size() == y.size());
  const auto n = a.size();
  if constexpr (std::is_same_v<Scalar, float>)
  {
    fsingle(n, a.data(), a.increment(), b.data(), b.increment(), y.data(), y.increment());
  }
  else
  {
    fdouble(n, a.data(), a.increment(), b.data(), b.increment(), y.data(), y.increment());
  }
};

// y := a + b
template <typename Scalar>
void vm_add(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsAddI, vdAddI);
};

// y := a - b
template <typename Scalar>
void vm_sub(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsSubI, vdSubI);
};

// y := apply(a, sqr)
template <typename Scalar>
void vm_sqr(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSqrI, vdSqrI);
};

// y := hadamard(a, b)
template <typename Scalar>
void vm_mul(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsMulI, vdMulI);
};

// y := apply(y, abs)
template <typename Scalar>
void vm_abs(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAbsI, vdAbsI);
};

// y[i] := (scalea * a[i] + shifta) / (scaleb * b[i] + shiftb)
template <typename Scalar>
void vm_linear_frac(const dense_vector_view<Scalar>& a,
                   const dense_vector_view<Scalar>& b,
                   Scalar scalea,
                   Scalar shifta,
                   Scalar scaleb,
                   Scalar shiftb,
                   dense_vector_view<Scalar>& y)
{
  assert(a.size() == b.size() && a.size() == y.size());
  const auto n = a.size();
  if constexpr (std::is_same_v<Scalar, float>)
  {
    vsLinearFracI(n, a.data(), a.increment(), b.data(), b.increment(), scalea, shifta, scaleb, shiftb, y.data(), y.increment());
  }
  else
  {
    vdLinearFracI(n, a.data(), a.increment(), b.data(), b.increment(), scalea, shifta, scaleb, shiftb, y.data(), y.increment());
  }
};

// y[i] := "element by element computation of the modulus function of vector a with respect to vector b" (???)
template <typename Scalar>
void vm_fmod(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsFmodI, vdFmodI);
};

// y := "element by element computation of the remainder function on the elements of vector a and the corresponding elements of vector b" (???)
template <typename Scalar>
void vm_remainder(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsRemainderI, vdRemainderI);
};

// y := apply(y, inverse)
template <typename Scalar>
void vm_inv(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsInvI, vdInvI);
};

// y := "element by element division of vector a by vector b"
template <typename Scalar>
void vm_div(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsDivI, vdDivI);
};

// y := apply(y, sqrt)
template <typename Scalar>
void vm_sqrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSqrtI, vdSqrtI);
};

// y := apply(y, inverse_sqrt)
template <typename Scalar>
void vm_inverse_sqrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsInvSqrtI, vdInvSqrtI);
};

// y := apply(y, cube_root)
template <typename Scalar>
void vm_cbrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCbrtI, vdCbrtI);
};

// y := apply(y, inverse_cube_root)
template <typename Scalar>
void vm_inverse_cbrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsInvCbrtI, vdInvCbrtI);
};

// y := apply(y, cbrt_square)
template <typename Scalar>
void vm_cbrt_square(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsPow2o3I, vdPow2o3I);
};

// y := apply(y, sqrt_cube)
template <typename Scalar>
void vm_sqrt_cube(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsPow3o2, vdPow3o2);
};

// y := "a to the power b for elements of two vectors"
template <typename Scalar>
void vm_pow(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsPowI, vdPowI);
};

// y := "vector a to the scalar power b"
template <typename Scalar>
void vm_powx(const dense_vector_view<Scalar>& a,
            Scalar b,
            dense_vector_view<Scalar>& y)
{
  assert(a.size() == y.size());
  const auto n = a.size();
  if constexpr (std::is_same_v<Scalar, float>)
  {
    vsPowxI(n, a.data(), a.increment(), b, y.data(), y.increment());
  }
  else
  {
    vdPowxI(n, a.data(), a.increment(), b, y.data(), y.increment());
  }
};

// y := "a to the power b for elements of two vectors, where the elements of vector argument a are all non-negative"
template <typename Scalar>
void vm_powr(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsPowrI, vdPowrI);
};

// y := "square root of sum of two squared elements"
template <typename Scalar>
void vm_hypot(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsHypotI, vdHypotI);
};

// y := apply(y, exp)
template <typename Scalar>
void vm_exp(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExpI, vdExpI);
};

// y := "base 2 exponential of vector elements"
template <typename Scalar>
void vm_exp2(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExp2I, vdExp2I);
};

// y := "base 10 exponential of vector elements"
template <typename Scalar>
void vm_exp10(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExp10I, vdExp10I);
};

// y := "exponential of vector elements decreased by 1"
template <typename Scalar>
void vm_expm1(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExpm1I, vdExpm1I);
};

// y := apply(y, ln)
template <typename Scalar>
void vm_ln(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLnI, vdLnI);
};

// y := apply(y, log2)
template <typename Scalar>
void vm_log2(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLog2I, vdLog2I);
};

// y := apply(y, log10)
template <typename Scalar>
void vm_log10(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLog10I, vdLog10I);
};

// y := "natural logarithm of vector elements that are increased by 1"
template <typename Scalar>
void vm_log1p(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLog1pI, vdLog1pI);
};

// y := "exponents of the elements of input vector a"
template <typename Scalar>
void vm_logb(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLogbI, vdLogbI);
};

// y := apply(y, cos)
template <typename Scalar>
void vm_cos(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCosI, vdCosI);
};

// y := apply(y, sin)
template <typename Scalar>
void vm_sin(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSinI, vdSinI);
};

// y, z := apply(y, sin), apply(y, cos)
template <typename Scalar>
void vm_sincos(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y, dense_vector_view<Scalar>& z)
{
  assert(a.size() == y.size() && a.size() == z.size());
  const auto n = a.size();
  if constexpr (std::is_same_v<Scalar, float>)
  {
    vsSinCosI(n, a.data(), a.increment(), y.data(), y.increment(), z.data(), z.increment());
  }
  else
  {
    vdSinCosI(n, a.data(), a.increment(), y.data(), y.increment(), z.data(), z.increment());
  }
};

// y := apply(y, tan)
template <typename Scalar>
void vm_tan(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTanI, vdTanI);
};

// y := apply(y, acos)
template <typename Scalar>
void vm_acos(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAcosI, vdAcosI);
};

// y := apply(y, asin)
template <typename Scalar>
void vm_asin(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAsinI, vdAsinI);
};

// y := apply(y, atan)
template <typename Scalar>
void vm_atan(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAtanI, vdAtanI);
};

// y := "cosine of vector elements multiplied by π"
template <typename Scalar>
void vm_cospi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCospiI, vdCospiI);
};

// y := "sine of vector elements multiplied by π"
template <typename Scalar>
void vm_sinpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSinpiI, vdSinpiI);
};

// y := "tangent of vector elements multiplied by π"
template <typename Scalar>
void vm_tanpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTanpiI, vdTanpiI);
};

// y := "inverse cosine of vector elements divided by π"
template <typename Scalar>
void vm_acospi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAcospiI, vdAcospiI);
};

// y := "inverse sine of vector elements divided by π"
template <typename Scalar>
void vm_asinpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAsinpiI, vdAsinpiI);
};

// y := "inverse tangent of vector elements divided by π"
template <typename Scalar>
void vm_atanpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAtanpiI, vdAtanpiI);
};

} // namespace nerva
