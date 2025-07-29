/* This module contains C++ helper routines for processing finufft
 * inputs and outputs. Its main purpose is to enable "kernel fusion",
 * i.e. do as much array processing as possible element-wise, instead
 * of array-wise as occurs in Numpy.
 */

#include <algorithm>
#include <complex>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include "cpu_helpers.hpp"
using cpu_helpers::NormKind;

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
void process_finufft_inputs(
   nifty_arr_1d<Scalar> t1_,
   nifty_arr_1d<Scalar> t2_,
   nifty_arr_2d<Complex<Scalar>> yw_,
   nifty_arr_2d<Complex<Scalar>> w_,
   nifty_arr_2d<Complex<Scalar>> w2_,
   nifty_arr_1d<Scalar> norm_,
   nifty_arr_1d<const Scalar> t_,
   nifty_arr_2d<const Scalar> y_,
   nifty_arr_2d<const Scalar> dy_,
   const Scalar fmin,
   const Scalar df,
   const size_t Nf,
   const bool center_data,
   const bool fit_mean,
   const bool psd_norm,
   int nthreads
) {
    auto t1   = t1_.view();
    auto t2   = t2_.view();
    auto yw   = yw_.view();
    auto w    = w_.view();
    auto w2   = w2_.view();
    auto norm = norm_.view();
    auto t    = t_.view();   // read-only
    auto y    = y_.view();   // read-only
    auto dy   = dy_.view();  // read-only

    size_t Nbatch = y.shape(0);
    size_t N      = y.shape(1);
    size_t Nshift = Nf / 2;

    bool broadcast_dy = dy.shape(1) == 1;

#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
    if (nthreads > omp_get_max_threads()) {
        fprintf(
           stderr,
           "[nifty-ls finufft] Warning: nthreads (%d) > omp_get_max_threads() (%d). Performance may be suboptimal.\n",
           nthreads,
           omp_get_max_threads()
        );
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
   reduction(vsum : wsum) reduction(vsum : yoff)
#endif
        for (size_t i = 0; i < Nbatch; ++i) {
            for (size_t j = 0; j < N; ++j) {
                w2(i, j) = 1 / (dy(i, j) * dy(i, j));
                if (center_data || fit_mean) { yoff[i] += w2(i, j).real() * y(i, j); }
                wsum[i] += w2(i, j).real();
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2) \
   reduction(vsum : yoff)
#endif
        for (size_t i = 0; i < Nbatch; ++i) {
            for (size_t j = 0; j < N; ++j) {
                w2(i, j) = 1.;
                if (center_data || fit_mean) { yoff[i] += y(i, j); }
            }
        }
    }

    // norm = (w2 * y**2).sum(axis=-1, keepdims=True)
    // We're storing w2 as complex
    // but it's real at this point in the code, before the phase shift
    // TODO: could try to taskify this loop
    for (size_t i = 0; i < Nbatch; ++i) {
        if (broadcast_dy) { wsum[i] = N; }
        if (psd_norm) {
            norm(i) = wsum[i];
        } else {
            norm(i) = 0;
        }
        if (center_data || fit_mean) { yoff[i] /= wsum[i]; }
        Scalar normi = norm(i);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) reduction(+ : normi)
#endif
        for (size_t j = 0; j < N; ++j) {
            w2(i, j).real(w2(i, j).real() / wsum[i]);  // w2 /= sum
            if (!psd_norm) {
                normi += w2(i, j).real() * (y(i, j) - yoff[i]) * (y(i, j) - yoff[i]);
            }
        }
        norm(i) = normi;
    }

    // if center_data or fit_mean:
    //     y = y - np.dot(w2, y)

    const std::complex<Scalar> phase_shift =
       std::complex<Scalar>(0, Nshift + fmin / df);
    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for (size_t j = 0; j < N; ++j) {
        t1(j) = TWO_PI * df * t(j);
        t2(j) = 2 * t1(j);

        for (size_t i = 0; i < Nbatch; ++i) {
            yw(i, j) = (y(i, j) - static_cast<Scalar>(yoff[i])) * w2(i, j)
                       * std::exp(phase_shift * t1(j));
            if (fit_mean) { w(i, j) = w2(i, j) * std::exp(phase_shift * t1(j)); }
            w2(i, j) *= std::exp(phase_shift * t2(j));
        }
    }
}

template <typename Scalar>
void process_finufft_outputs(
   nifty_arr_2d<Scalar> power_,
   nifty_arr_2d<const std::complex<Scalar>> f1_,
   nifty_arr_2d<const std::complex<Scalar>> fw_,
   nifty_arr_2d<const std::complex<Scalar>> f2_,
   nifty_arr_1d<const Scalar> norm_YY_,
   const NormKind norm_kind,
   const bool fit_mean,
   int nthreads
) {
    auto power   = power_.view();
    auto f1      = f1_.view();       // read-only
    auto fw      = fw_.view();       // read-only
    auto f2      = f2_.view();       // read-only
    auto norm_YY = norm_YY_.view();  // read-only

    size_t Nbatch = f1.shape(0);
    size_t N      = f1.shape(1);

    const Scalar SQRT_HALF = std::sqrt(0.5);

#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
#else
    (void) nthreads;  // suppress unused variable warning
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2)
#endif
    for (size_t i = 0; i < Nbatch; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Scalar tan_2omega_tau;
            if (fit_mean) {
                tan_2omega_tau =
                   (f2(i, j).imag() - 2 * fw(i, j).imag() * fw(i, j).real())
                   / (f2(i, j).real()
                      - (fw(i, j).real() * fw(i, j).real()
                         - fw(i, j).imag() * fw(i, j).imag()));
            } else {
                tan_2omega_tau = f2(i, j).imag() / f2(i, j).real();
            }
            Scalar S2w =
               tan_2omega_tau / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar C2w = 1 / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar Cw  = SQRT_HALF * std::sqrt(1 + C2w);
            Scalar Sw  = SQRT_HALF * (S2w >= 0 ? 1 : -1) * std::sqrt(1 - C2w);

            Scalar YC = f1(i, j).real() * Cw + f1(i, j).imag() * Sw;
            Scalar YS = f1(i, j).imag() * Cw - f1(i, j).real() * Sw;
            Scalar CC = (1 + f2(i, j).real() * C2w + f2(i, j).imag() * S2w) / 2;
            Scalar SS = (1 - f2(i, j).real() * C2w - f2(i, j).imag() * S2w) / 2;

            if (fit_mean) {
                Scalar CC_fac = fw(i, j).real() * Cw + fw(i, j).imag() * Sw;
                Scalar SS_fac = fw(i, j).imag() * Cw - fw(i, j).real() * Sw;
                CC -= CC_fac * CC_fac;
                SS -= SS_fac * SS_fac;
            }

            power(i, j) = YC * YC / CC + YS * YS / SS;

            switch (norm_kind) {
                case NormKind::Standard:
                    power(i, j) /= norm_YY(i);
                    break;
                case NormKind::Model:
                    power(i, j) /= norm_YY(i) - power(i, j);
                    break;
                case NormKind::Log:
                    power(i, j) = -std::log(1 - power(i, j) / norm_YY(i));
                    break;
                case NormKind::PSD:
                    // For PSD, norm_YY is actually (dy ** -2).sum()
                    // instead of (w2 * y**2).sum()
                    power(i, j) *= 0.5 * norm_YY(i);
                    break;
            }
        }
    }
}

template <typename Scalar>
Scalar _normalization(
   nifty_arr_1d<const Scalar> y_, std::vector<Scalar> &w, Scalar wsum, Scalar yoff
) {
    // only used for winding

    const auto y = y_.view();  // read-only

    Scalar invnorm = Scalar(0);
    const size_t N = y.shape(0);
    for (size_t n = 0; n < N; ++n) {
        invnorm += (w[n] / wsum) * (y(n) - yoff) * (y(n) - yoff);
    }
    return Scalar(1) / invnorm;
}

template <typename Scalar>
void compute_winding(
   nifty_arr_1d<Scalar> power_,
   nifty_arr_1d<const Scalar> t_,
   nifty_arr_1d<const Scalar> y_,
   nifty_arr_1d<const Scalar> dy_,
   const Scalar fmin,
   const Scalar df,
   const bool center_data,
   const bool fit_mean
) {

    /* Use the "phase winding" method to compute the periodogram.
     * It's faster than naive sin/cos evaluation, but much slower than finufft.
     * It uses trigonometric identities rather than approximations,
     * and is only used for testing purposes.
     */

    auto sgn = [](Scalar x) -> Scalar { return (x > 0) - (x < 0); };

    auto power = power_.view();
    auto t     = t_.view();   // read-only
    auto y     = y_.view();   // read-only
    auto dy    = dy_.view();  // read-only

    const size_t N  = y.shape(0);
    const size_t Nf = power.shape(0);

    std::vector<Scalar> w(N);

    Scalar wsum = 0.0;
    Scalar yoff = 0.0;
    for (size_t n = 0; n < N; ++n) {
        w[n] = 1 / (dy(n) * dy(n));
        wsum += w[n];
        if (center_data || fit_mean) { yoff += w[n] * y(n); }
    }

    if (center_data || fit_mean) { yoff /= wsum; }

    const Scalar sqrt_half = std::sqrt(Scalar(0.5));
    const Scalar domega    = 2 * M_PI * df;
    const Scalar omega0    = 2 * M_PI * fmin;

    const Scalar norm = _normalization(y_, w, wsum, yoff);

    Scalar *sin_omegat  = new Scalar[N];
    Scalar *cos_omegat  = new Scalar[N];
    Scalar *sin_domegat = new Scalar[N];
    Scalar *cos_domegat = new Scalar[N];
    for (size_t n = 0; n < N; n++) {
        Scalar omega0t = omega0 * t(n);
        Scalar domegat = domega * t(n);
        sin_omegat[n]  = std::sin(omega0t);
        cos_omegat[n]  = std::cos(omega0t);
        sin_domegat[n] = std::sin(domegat);
        cos_domegat[n] = std::cos(domegat);
    }

    for (size_t m = 0; m < Nf; ++m) {
        Scalar S = 0., C = 0.;
        Scalar Sh = 0., Ch = 0.;
        Scalar S2 = 0., C2 = 0.;

        for (size_t n = 0; n < N; ++n) {
            Scalar wn  = w[n] / wsum;
            Scalar hn  = wn * (y(n) - yoff);
            Scalar sin = sin_omegat[n];
            Scalar cos = cos_omegat[n];

            Sh += hn * sin;
            Ch += hn * cos;

            if (fit_mean) {
                S += wn * sin;
                C += wn * cos;
            }

            // sin(2*x) = 2 * sin(x) * cos(x)
            // cos(2*x) = cos(x)*cos(x) - sin(x)*sin(x)
            S2 += 2 * wn * sin * cos;
            C2 += wn * (cos * cos - sin * sin);

            // sin(x + dx) = sin(x) cos(dx) + cos(x) sin(dx)
            // cos(x + dx) = cos(x) cos(dx) - sin(x) sin(dx)
            sin_omegat[n] =
               sin_omegat[n] * cos_domegat[n] + cos_omegat[n] * sin_domegat[n];
            cos_omegat[n] = cos_omegat[n] * cos_domegat[n] - sin * sin_domegat[n];
        }

        Scalar tan_2omega_tau;
        if (fit_mean) {
            tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S));
        } else {
            tan_2omega_tau = S2 / C2;
        }

        Scalar C2w = Scalar(1) / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
        Scalar S2w = tan_2omega_tau * C2w;
        Scalar Cw  = sqrt_half * std::sqrt(1 + C2w);
        Scalar Sw  = sqrt_half * sgn(S2w) * std::sqrt(1 - C2w);

        Scalar YC = Ch * Cw + Sh * Sw;
        Scalar YS = Sh * Cw - Ch * Sw;
        Scalar CC = 0.5 * (1 + C2 * C2w + S2 * S2w);
        Scalar SS = 0.5 * (1 - C2 * C2w - S2 * S2w);

        if (fit_mean) {
            Scalar CC_fac = C * Cw + S * Sw;
            Scalar SS_fac = S * Cw - C * Sw;
            CC -= CC_fac * CC_fac;
            SS -= SS_fac * SS_fac;
        }

        power(m) = norm * (YC * YC / CC + YS * YS / SS);
    }

    delete[] sin_omegat;
    delete[] cos_omegat;
    delete[] sin_domegat;
    delete[] cos_domegat;
}

NB_MODULE(cpu_helpers, m) {
    // We're using noconvert() here to ensure the input arrays are not copied

    m.def(
       "process_finufft_inputs",
       &process_finufft_inputs<double>,
       "t1"_a.noconvert(),
       "t2"_a.noconvert(),
       "yw"_a.noconvert(),
       "w"_a.noconvert(),
       "w2"_a.noconvert(),
       "norm"_a.noconvert(),
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "center_data"_a,
       "fit_mean"_a,
       "psd_normalization"_a,
       "nthreads"_a
    );

    m.def(
       "process_finufft_inputs",
       &process_finufft_inputs<float>,
       "t1"_a.noconvert(),
       "t2"_a.noconvert(),
       "yw"_a.noconvert(),
       "w"_a.noconvert(),
       "w2"_a.noconvert(),
       "norm"_a.noconvert(),
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "center_data"_a,
       "fit_mean"_a,
       "psd_normalization"_a,
       "nthreads"_a
    );

    m.def(
       "process_finufft_outputs",
       &process_finufft_outputs<double>,
       "power"_a.noconvert(),
       "f1"_a.noconvert(),
       "fw"_a.noconvert(),
       "f2"_a.noconvert(),
       "norm_YY"_a.noconvert(),
       "norm_kind"_a,
       "fit_mean"_a,
       "nthreads"_a
    );

    m.def(
       "process_finufft_outputs",
       &process_finufft_outputs<float>,
       "power"_a.noconvert(),
       "f1"_a.noconvert(),
       "fw"_a.noconvert(),
       "f2"_a.noconvert(),
       "norm_YY"_a.noconvert(),
       "norm_kind"_a,
       "fit_mean"_a,
       "nthreads"_a
    );

    m.def(
       "compute_winding",
       &compute_winding<double>,
       "power"_a.noconvert(),
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "w"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "center_data"_a,
       "fit_mean"_a
    );

    nb::enum_<NormKind>(m, "NormKind")
       .value("Standard", NormKind::Standard)
       .value("Model", NormKind::Model)
       .value("Log", NormKind::Log)
       .value("PSD", NormKind::PSD);
}
