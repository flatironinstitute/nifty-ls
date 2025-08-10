#ifndef UTILS_HELPERS_HPP
#define UTILS_HELPERS_HPP

#include <algorithm>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

const double PI = 3.14159265358979323846;

namespace nb = nanobind;
using namespace nb::literals;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_3d = nb::ndarray<Scalar, nb::ndim<3>, nb::device::cpu>;

template <typename Scalar>
using Complex = std::complex<Scalar>;

#ifdef _OPENMP
#include <omp.h>
// Declare a reduction for std::vector<double> using std::transform
#pragma omp declare reduction(                                \
      vsum : std::vector<double> : std::transform(            \
            omp_out.begin(),                                  \
               omp_out.end(),                                 \
               omp_in.begin(),                                \
               omp_out.begin(),                               \
               std::plus<double>()                            \
      )                                                       \
) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
#endif

namespace utils_helpers {
    enum class NormKind {
        Standard,
        Model,
        Log,
        PSD
    };

    enum class TermType {
        Sine,
        Cosine
    };
}  // namespace utils_helpers

#endif  // UTILS_HELPERS_HPP
