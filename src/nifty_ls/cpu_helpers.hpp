/* This module contains C++ helper routines for processing finufft
 * inputs and outputs.
 */

#include <algorithm>
#include <complex>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#ifdef _OPENMP
#include <omp.h>

#include "utils_helpers.hpp"
using utils_helpers::NormKind;

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

namespace nb = nanobind;
using namespace nb::literals;

const double PI = 3.14159265358979323846;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

template <typename Scalar>
using Complex = std::complex<Scalar>;

template <typename Scalar>
void process_finufft_inputs_raw(
   Scalar *t1,               // (N)
   Scalar *t2,               // (N)
   Complex<Scalar> *yw,      // (Nbatch, N)
   Complex<Scalar> *w,       // (Nbatch, N)
   Complex<Scalar> *w2,      // (Nbatch, N)
   Scalar *norm,             // (Nbatch)
   const Scalar *t,          // input, (N)
   const Scalar *y,          // input, (Nbatch, N)
   const Scalar *dy,         // input, (Nbatch, N)
   const bool broadcast_dy,  // input
   const Scalar fmin,
   const Scalar df,
   const size_t Nf,
   const bool center_data,
   const bool fit_mean,
   const bool psd_norm,
   int nthreads,
   const size_t Nbatch,
   const size_t N
) {
    size_t Nshift = Nf / 2;

#ifdef _OPENMP
    if (nthreads != 1) {
        if (nthreads < 1) { nthreads = omp_get_max_threads(); }
        if (nthreads > omp_get_max_threads()) {
            fprintf(
               stderr,
               "[nifty-ls finufft_heterobatch] Warning: nthreads (%d) > omp_get_max_threads() (%d). Performance may be suboptimal.\n",
               nthreads,
               omp_get_max_threads()
            );
        }
    }
#else
    (void) nthreads;  // suppress unused variable warning
#endif

    // w2 = dy**-2.
    std::vector<double> wsum(Nbatch, 0.);  // use double for stability
    std::vector<double> yoff(Nbatch, 0.);

    if (!broadcast_dy) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2) \
   reduction(vsum : wsum) reduction(vsum : yoff) if (nthreads > 1)
#endif
        // w2 = dy**-2.
        for (size_t i = 0; i < Nbatch; ++i) {
            for (size_t j = 0; j < N; ++j) {
                w2[i * N + j] = 1 / (dy[i * N + j] * dy[i * N + j]);
                if (center_data || fit_mean) {
                    yoff[i] += w2[i * N + j].real() * y[i * N + j];
                }
                wsum[i] += w2[i * N + j].real();
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2) \
   reduction(vsum : yoff) if (nthreads > 1)
#endif
        for (size_t i = 0; i < Nbatch; ++i) {
            for (size_t j = 0; j < N; ++j) {
                w2[i * N + j] = Scalar(1);
                if (center_data || fit_mean) { yoff[i] += y[i * N + j]; }
            }
        }
    }
    // Not taskified — dynamic scheduling overhead outweighs benefits
    //  norm = (w2 * y**2).sum(axis=-1)
    for (size_t i = 0; i < Nbatch; ++i) {
        if (broadcast_dy) { wsum[i] = N; }
        if (psd_norm) {
            norm[i] = wsum[i];
        } else {
            norm[i] = Scalar(0);
        }
        if (center_data || fit_mean) { yoff[i] /= wsum[i]; }
        Scalar normi = norm[i];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
   reduction(+ : normi) if (nthreads > 1)
#endif
        for (size_t j = 0; j < N; ++j) {
            w2[i * N + j].real(w2[i * N + j].real() / wsum[i]);  // w2 /= sum
            if (!psd_norm) {
                normi += w2[i * N + j].real() * (y[i * N + j] - yoff[i])
                         * (y[i * N + j] - yoff[i]);
            }
        }
        norm[i] = normi;
    }

    // Phase shift
    const Complex<Scalar> phase_shift(0, Nshift + fmin / df);
    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static) if (nthreads > 1)
#endif
    for (size_t j = 0; j < N; ++j) {
        t1[j] = TWO_PI * df * t[j];
        t2[j] = 2 * t1[j];
        for (size_t i = 0; i < Nbatch; ++i) {
            yw[i * N + j] = (y[i * N + j] - static_cast<Scalar>(yoff[i]))
                            * w2[i * N + j] * std::exp(phase_shift * t1[j]);
            if (fit_mean) {
                w[i * N + j] = w2[i * N + j] * std::exp(phase_shift * t1[j]);
            }
            w2[i * N + j] *= std::exp(phase_shift * t2[j]);
        }
    }
}

template <typename Scalar>
void process_finufft_outputs_raw(
   Scalar *power,              // f1.shape, (Nbatch, N)
   const Complex<Scalar> *f1,  // (Nbatch, Nf)
   const Complex<Scalar> *fw,  // (Nbatch, Nf)
   const Complex<Scalar> *f2,  // (Nbatch, Nf)
   const Scalar *norm_YY,      // (Nbatch,)
   const NormKind norm_kind,
   const bool fit_mean,
   int nthreads,
   const size_t Nbatch,
   const size_t N
) {
    const Scalar SQRT_HALF = std::sqrt(0.5);

#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
#else
    (void) nthreads;  // suppress unused variable warning
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
   collapse(2) if (nthreads > 1)
#endif
    for (size_t i = 0; i < Nbatch; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Scalar tan_2omega_tau;
            if (fit_mean) {
                tan_2omega_tau = (f2[i * N + j].imag()
                                  - 2 * fw[i * N + j].imag() * fw[i * N + j].real())
                                 / (f2[i * N + j].real()
                                    - (fw[i * N + j].real() * fw[i * N + j].real()
                                       - fw[i * N + j].imag() * fw[i * N + j].imag()));
            } else {
                tan_2omega_tau = f2[i * N + j].imag() / f2[i * N + j].real();
            }
            Scalar S2w =
               tan_2omega_tau / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar C2w = 1 / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar Cw  = SQRT_HALF * std::sqrt(1 + C2w);
            Scalar Sw  = SQRT_HALF * (S2w >= 0 ? 1 : -1) * std::sqrt(1 - C2w);

            Scalar YC = f1[i * N + j].real() * Cw + f1[i * N + j].imag() * Sw;
            Scalar YS = f1[i * N + j].imag() * Cw - f1[i * N + j].real() * Sw;
            Scalar CC =
               (1 + f2[i * N + j].real() * C2w + f2[i * N + j].imag() * S2w) / 2;
            Scalar SS =
               (1 - f2[i * N + j].real() * C2w - f2[i * N + j].imag() * S2w) / 2;

            if (fit_mean) {
                Scalar CC_fac = fw[i * N + j].real() * Cw + fw[i * N + j].imag() * Sw;
                Scalar SS_fac = fw[i * N + j].imag() * Cw - fw[i * N + j].real() * Sw;
                CC -= CC_fac * CC_fac;
                SS -= SS_fac * SS_fac;
            }

            power[i * N + j] = YC * YC / CC + YS * YS / SS;

            switch (norm_kind) {
                case NormKind::Standard:
                    power[i * N + j] /= norm_YY[i];
                    break;
                case NormKind::Model:
                    power[i * N + j] /= norm_YY[i] - power[i * N + j];
                    break;
                case NormKind::Log:
                    power[i * N + j] = -std::log(1 - power[i * N + j] / norm_YY[i]);
                    break;
                case NormKind::PSD:
                    // For PSD, norm_YY is actually (dy ** -2).sum()
                    // instead of (w2 * y**2).sum()
                    power[i * N + j] *= 0.5 * norm_YY[i];
                    break;
            }
        }
    }
}
