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
#include <cassert>
#include <type_traits>

#ifdef NERVA_SYCL
#include <sycl/sycl.hpp>
#endif

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

    Scalar& operator()(long i)
    {
      return m_data[i * m_increment];
    }

    Scalar operator[](long i) const
    {
      return m_data[i * m_increment];
    }

    Scalar& operator[](long i)
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
//                     Vector Mathematical Functions
//----------------------------------------------------------------------//
// See https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/vm-mathematical-functions.html

//--- helper functions ---//
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

//--- Arithmetic Functions ---//

// Adds vector elements
template <typename Scalar>
void vm_add(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsAddI, vdAddI);
};

// Subtracts vector elements
template <typename Scalar>
void vm_sub(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsSubI, vdSubI);
};

// Squares vector elements
template <typename Scalar>
void vm_sqr(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSqrI, vdSqrI);
};

// Multiplies vector elements
template <typename Scalar>
void vm_mul(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsMulI, vdMulI);
};

// Computes the absolute value of vector elements
template <typename Scalar>
void vm_abs(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAbsI, vdAbsI);
};

// Performs linear fraction transformation of vectors
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

// Performs element by element computation of the modulus function of vector a with respect to vector b
template <typename Scalar>
void vm_fmod(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsFmodI, vdFmodI);
};

// Performs element by element computation of the remainder function on the elements of vector a and the corresponding elements of vector b
template <typename Scalar>
void vm_remainder(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsRemainderI, vdRemainderI);
};

//--- Power and Root Functions ---//

// Inverts vector elements
template <typename Scalar>
void vm_inv(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsInvI, vdInvI);
};

// Divides elements of one vector by elements of the second vector
template <typename Scalar>
void vm_div(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsDivI, vdDivI);
};

// Computes the square root of vector elements
template <typename Scalar>
void vm_sqrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSqrtI, vdSqrtI);
};

// Computes the inverse square root of vector elements
template <typename Scalar>
void vm_inverse_sqrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsInvSqrtI, vdInvSqrtI);
};

// Computes the cube root of vector elements
template <typename Scalar>
void vm_cbrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCbrtI, vdCbrtI);
};

// Computes the inverse cube root of vector elements
template <typename Scalar>
void vm_inverse_cbrt(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsInvCbrtI, vdInvCbrtI);
};

// Computes the cube root of the square of each vector element
template <typename Scalar>
void vm_cbrt_square(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsPow2o3I, vdPow2o3I);
};

// Computes the square root of the cube of each vector element
template <typename Scalar>
void vm_sqrt_cube(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsPow3o2, vdPow3o2);
};

// Raises each vector element to the specified power
template <typename Scalar>
void vm_pow(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsPowI, vdPowI);
};

// Raises each vector element to the constant power
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

// Computes a to the power b for elements of two vectors, where the elements of vector argument a are all non-negative
template <typename Scalar>
void vm_powr(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsPowrI, vdPowrI);
};

// Computes the square root of sum of squares
template <typename Scalar>
void vm_hypot(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsHypotI, vdHypotI);
};

//--- Exponential and Logarithmic Functions ---//

// Computes the base e exponential of vector elements
template <typename Scalar>
void vm_exp(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExpI, vdExpI);
};

// Computes the base 2 exponential of vector elements
template <typename Scalar>
void vm_exp2(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExp2I, vdExp2I);
};

// Computes the base 10 exponential of vector elements
template <typename Scalar>
void vm_exp10(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExp10I, vdExp10I);
};

// Computes the base e exponential of vector elements decreased by 1
template <typename Scalar>
void vm_expm1(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExpm1I, vdExpm1I);
};

// Computes the natural logarithm of vector elements
template <typename Scalar>
void vm_ln(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLnI, vdLnI);
};

// Computes the base 2 logarithm of vector elements
template <typename Scalar>
void vm_log2(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLog2I, vdLog2I);
};

// Computes the base 10 logarithm of vector elements
template <typename Scalar>
void vm_log10(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLog10I, vdLog10I);
};

// Computes the natural logarithm of vector elements that are increased by 1
template <typename Scalar>
void vm_log1p(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLog1pI, vdLog1pI);
};

// Computes the exponents of the elements of input vector a
template <typename Scalar>
void vm_logb(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLogbI, vdLogbI);
};

//--- Trigonometric Functions ---//

// Computes the cosine of vector elements
template <typename Scalar>
void vm_cos(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCosI, vdCosI);
};

// Computes the sine of vector elements
template <typename Scalar>
void vm_sin(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSinI, vdSinI);
};

// Computes the sine and cosine of vector elements
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

// Computes the tangent of vector elements
template <typename Scalar>
void vm_tan(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTanI, vdTanI);
};

// Computes the inverse cosine of vector elements
template <typename Scalar>
void vm_acos(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAcosI, vdAcosI);
};

// Computes the inverse sine of vector elements
template <typename Scalar>
void vm_asin(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAsinI, vdAsinI);
};

// Computes the inverse tangent of vector elements
template <typename Scalar>
void vm_atan(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAtanI, vdAtanI);
};

// Computes the four-quadrant inverse tangent of ratios of the elements of two vectors
template <typename Scalar>
void vm_atan2(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsAtan2I, vsAtan2I);
};

// Computes the cosine of vector elements multiplied by π
template <typename Scalar>
void vm_cospi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCospiI, vdCospiI);
};

// Computes the sine of vector elements multiplied by π
template <typename Scalar>
void vm_sinpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSinpiI, vdSinpiI);
};

// Computes the tangent of vector elements multiplied by π
template <typename Scalar>
void vm_tanpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTanpiI, vdTanpiI);
};

// Computes the inverse cosine of vector elements divided by π
template <typename Scalar>
void vm_acospi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAcospiI, vdAcospiI);
};

// Computes the inverse sine of vector elements divided by π
template <typename Scalar>
void vm_asinpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAsinpiI, vdAsinpiI);
};

// Computes the inverse tangent of vector elements divided by π
template <typename Scalar>
void vm_atanpi(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAtanpiI, vdAtanpiI);
};

// Computes the four-quadrant inverse tangent of the ratios of the corresponding elementss of two vectors divided by π
template <typename Scalar>
void vm_atan2pi(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsAtan2piI, vdAtan2piI);
};

// Computes the cosine of vector elements multiplied by π/180
template <typename Scalar>
void vm_cosd(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCosdI, vdCosdI);
};

// Computes the sine of vector elements multiplied by π/180
template <typename Scalar>
void vm_sind(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSindI, vdSindI);
};

// Computes the tangent of vector elements multiplied by π/180
template <typename Scalar>
void vm_tand(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTandI, vdTandI);
};

//--- Hyperbolic Functions ---//

// Computes the hyperbolic cosine of vector elements
template <typename Scalar>
void vm_cosh(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCoshI, vdCoshI);
};

// Computes the hyperbolic sine of vector elements
template <typename Scalar>
void vm_sinh(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsSinhI, vdSinhI);
};

// Computes the hyperbolic tangent of vector elements
template <typename Scalar>
void vm_tanh(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTanhI, vdTanhI);
};

// Computes the inverse hyperbolic cosine of vector elements
template <typename Scalar>
void vm_acosh(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAcoshI, vdAcoshI);
};

// Computes the inverse hyperbolic sine of vector elements
template <typename Scalar>
void vm_asinh(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAsinhI, vdAsinhI);
};

// Computes the inverse hyperbolic tangent of vector elements.
template <typename Scalar>
void vm_atanh(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsAtanhI, vdAtanhI);
};

//--- Special Functions ---//

// Computes the error function value of vector elements
template <typename Scalar>
void vm_erf(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsErfI, vdErfI);
};

// Computes the complementary error function value of vector elements
template <typename Scalar>
void vm_erfc(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsErfcI, vdErfcI);
};

// Computes the cumulative normal distribution function value of vector elements
template <typename Scalar>
void vm_cdfnorm(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCdfNormI, vdCdfNormI);
};

// Computes the inverse error function value of vector elements
template <typename Scalar>
void vm_erfinv(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsErfInvI, vdErfInvI);
};

// Computes the inverse complementary error function value of vector elements
template <typename Scalar>
void vm_erfcinv(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsErfcInvI, vdErfcInvI);
};

// Computes the inverse cumulative normal distribution function value of vector elements
template <typename Scalar>
void vm_cdfnorminv(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCdfNormInvI, vdCdfNormInvI);
};

// Computes the natural logarithm for the absolute value of the gamma function of vector elements
template <typename Scalar>
void vm_lgamma(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsLGammaI, vdLGammaI);
};

// Computes the gamma function of vector elements
template <typename Scalar>
void vm_tgamma(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTGammaI, vdTGammaI);
};

// Computes the exponential integral of vector elements
template <typename Scalar>
void vm_expint(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsExpInt1I, vdExpInt1I);
};

//--- Rounding Functions ---//

// Rounds towards minus infinity
template <typename Scalar>
void vm_floor(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsFloorI, vdFloorI);
};

// Rounds towards plus infinity
template <typename Scalar>
void vm_XXX(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsCeilI, vdCeilI);
};

// Rounds towards zero infinity
template <typename Scalar>
void vm_trunc(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsTruncI, vdTruncI);
};

// Rounds to nearest integer
template <typename Scalar>
void vm_round(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsRoundI, vdRoundI);
};

// Rounds according to current mode
template <typename Scalar>
void vm_nearby_int(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsNearbyIntI, vdNearbyIntI);
};

// Rounds according to current mode and raising inexact result exception
template <typename Scalar>
void vm_rint(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsRintI, vdRintI);
};

// Computes the integer and fractional parts
template <typename Scalar>
void vm_modf(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y, dense_vector_view<Scalar>& z)
{
  assert(a.size() == y.size() && a.size() == z.size());
  const auto n = a.size();
  if constexpr (std::is_same_v<Scalar, float>)
  {
    vsModfI(n, a.data(), a.increment(), y.data(), y.increment(), z.data(), z.increment());
  }
  else
  {
    vdModfI(n, a.data(), a.increment(), y.data(), y.increment(), z.data(), z.increment());
  }
};

// Computes the fractional part
template <typename Scalar>
void vm_frac(const dense_vector_view<Scalar>& a, dense_vector_view<Scalar>& y)
{
  vm_apply_unary_operation(a, y, vsFracI, vdFracI);
};

//--- Miscellaneous Functions ---//

// Returns vector of elements of one argument with signs changed to match other argument elements
template <typename Scalar>
void vm_copy_sign(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsCopySignI, vdCopySignI);
};

// Returns vector of elements containing the next representable floating-point values following the values from the elements of one vector in the direction of the corresponding elements of another vector
template <typename Scalar>
void vm_next_after(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsNextAfterI, vdNextAfterI);
};

// Returns vector containing the differences of the corresponding elements of the vector arguments if the first is larger and +0 otherwise
template <typename Scalar>
void vm_fdim(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsFdimI, vdFdimI);
};

// Returns the larger of each pair of elements of the two vector arguments
template <typename Scalar>
void vm_fmax(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsFmaxI, vdFmaxI);
};

// Returns the smaller of each pair of elements of the two vector arguments
template <typename Scalar>
void vm_fmin(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsFminI, vdFminI);
};

// Returns the element with the larger magnitude between each pair of elements of the two vector arguments
template <typename Scalar>
void vm_maxmag(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsMaxMagI, vdMaxMagI);
};

// Returns the element with the smaller magnitude between each pair of elements of the two vector arguments
template <typename Scalar>
void vm_minmag(const dense_vector_view<Scalar>& a, const dense_vector_view<Scalar>& b, dense_vector_view<Scalar>& y)
{
  vm_apply_binary_operation(a, b, y, vsMinMagI, vdMinMagI);
};

//----------------------------------------------------------------------//
//                     SYCL implementations
//----------------------------------------------------------------------//

#ifdef NERVA_SYCL
// z := a * x + b * y
template <typename Scalar>
void assign_axby(Scalar a, Scalar b, const dense_vector_view<Scalar>& x, const dense_vector_view<Scalar>& y, dense_vector_view<Scalar>& z)
{
  sycl::queue q;
  sycl::buffer<Scalar, 1> x_buffer{ const_cast<Scalar*>(x.data()), sycl::range<1>{static_cast<std::size_t>(x.size())}};
  sycl::buffer<Scalar, 1> y_buffer{ const_cast<Scalar*>(y.data()), sycl::range<1>{static_cast<std::size_t>(y.size())}};
  sycl::buffer<Scalar, 1> z_buffer{ z.data(), sycl::range<1>{static_cast<std::size_t>(z.size())}};

  q.submit([&](sycl::handler& cgh)
  {
    auto x = x_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto y = y_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto z = z_buffer.template get_access<sycl::access::mode::write>(cgh);
    std::size_t n = x.size();

    cgh.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i)
    {
      z[i] = a * x[i] + b * y[i];
    });
  }).wait();
};

// z := z + a * x + b * y
template <typename Scalar, typename Buffer>
void add_axby(Scalar a, Scalar b, Buffer& x, Buffer& y, Buffer& z)
{
  sycl::queue q;
  sycl::buffer<Scalar, 1> x_buffer{ const_cast<Scalar*>(x.data()), sycl::range<1>{static_cast<std::size_t>(x.size())}};
  sycl::buffer<Scalar, 1> y_buffer{ const_cast<Scalar*>(y.data()), sycl::range<1>{static_cast<std::size_t>(y.size())}};
  sycl::buffer<Scalar, 1> z_buffer{ z.data(), sycl::range<1>{static_cast<std::size_t>(z.size())}};

  q.submit([&](sycl::handler& cgh)
  {
    auto x = x_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto y = y_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto z = z_buffer.template get_access<sycl::access::mode::write>(cgh);
    std::size_t n = x.size();

    cgh.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i)
    {
      z[i] += a * x[i] + b * y[i];
    });
  }).wait();
};
#endif

} // namespace nerva
