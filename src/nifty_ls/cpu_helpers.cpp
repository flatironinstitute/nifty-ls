/* This module contains C++ helper routines for processing finufft
 * inputs and outputs. Its main purpose is to enable "kernel fusion",
 * i.e. do as much array processing as possible element-wise, instead
 * of array-wise as occurs in Numpy.
*/

#include <complex>
#include <algorithm>
#include <vector>

#include <omp.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

// Declare a reduction for std::vector<double> using std::transform
#pragma omp declare reduction( \
        vsum : \
        std::vector<double> : \
        std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) \
    ) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

namespace nb = nanobind;
using namespace nb::literals;

const double PI = 3.14159265358979323846;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

template <typename Scalar>
using Complex = std::complex<Scalar>;

enum class NormKind {
    Standard,
    Model,
    Log,
    PSD
};

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
    auto t1 = t1_.view();
    auto t2 = t2_.view();
    auto yw = yw_.view();
    auto w = w_.view();
    auto w2  = w2_ .view();
    auto norm = norm_.view();
    auto t  = t_ .view();  // read-only
    auto y  = y_ .view();  // read-only
    auto dy = dy_.view();  // read-only

    size_t Nbatch = y.shape(0);
    size_t N = y.shape(1);
    size_t Nshift = Nf / 2;

    if (nthreads < 1){
        nthreads = omp_get_max_threads();
    }
    omp_set_num_threads(nthreads);

    // w2 = dy**-2.
    std::vector<double> wsum(Nbatch, 0.);  // use double for stability
    std::vector<double> yoff(Nbatch, 0.);
    
    #pragma omp parallel for schedule(static) collapse(2) reduction(vsum:wsum) reduction(vsum:yoff)
    for (size_t i = 0; i < Nbatch; ++i) {
        for (size_t j = 0; j < N; ++j) {
            w2(i, j) = 1 / (dy(i, j) * dy(i, j));
            if (center_data || fit_mean){
                yoff[i] += w2(i, j).real() * y(i, j);
            }
            wsum[i] += w2(i, j).real();
        }
    }

    // norm = (w2 * y**2).sum(axis=-1, keepdims=True)
    // We're storing w2 as complex
    // but it's real at this point in the code, before the phase shift
    // TODO: could try to taskify this loop
    for (size_t i = 0; i < Nbatch; ++i) {
        if (psd_norm) {
            norm(i) = wsum[i];
        } else {
            norm(i) = 0;
        }
        if(center_data || fit_mean) {
            yoff[i] /= wsum[i];
        }
        Scalar normi = norm(i);
        #pragma omp parallel for schedule(static) reduction(+:normi)
        for (size_t j = 0; j < N; ++j) {
            w2(i, j).real( w2(i, j).real() / wsum[i] );  // w2 /= sum
            if (!psd_norm) {
                normi += w2(i, j).real() * (y(i, j) - yoff[i]) * (y(i, j) - yoff[i]);
            }
        }
        norm(i) = normi;
    }

    // if center_data or fit_mean:
    //     y = y - np.dot(w2, y)

    const std::complex<Scalar> phase_shift = std::complex<Scalar>(0, Nshift + fmin/df);
    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);
    #pragma omp parallel for schedule(static)
    for(size_t j = 0; j < N; ++j) {
        t1(j) = TWO_PI * df * t(j);
        t2(j) = 2 * t1(j);

        for(size_t i = 0; i < Nbatch; ++i) {
            yw(i, j) = (y(i, j) - static_cast<Scalar>(yoff[i])) * w2(i, j) * std::exp(phase_shift * t1(j));
            if (fit_mean){
                w(i, j) = w2(i, j) * std::exp(phase_shift * t1(j));
            }
            w2(i, j) *= std::exp(phase_shift * t2(j));
        }

        // TODO: if fmin/df is an integer, we may be able to do
        //   t = ((df * t) % 1) * 2 * np.pi
        // Could help a lot with range-reduction performance.
        t1(j) = std::fmod(t1(j), TWO_PI);
        t2(j) = std::fmod(t2(j), TWO_PI);
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
){
    auto power = power_.view();
    auto f1 = f1_.view();  // read-only
    auto fw = fw_.view();  // read-only
    auto f2 = f2_.view();  // read-only
    auto norm_YY = norm_YY_.view();  // read-only

    size_t Nbatch = f1.shape(0);
    size_t N = f1.shape(1);

    const Scalar SQRT_HALF = std::sqrt(0.5);

    if (nthreads < 1){
        nthreads = omp_get_max_threads();
    }
    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static) collapse(2)
    for(size_t i = 0; i < Nbatch; ++i) {
        for(size_t j = 0; j < N; ++j) {
            Scalar tan_2omega_tau;
            if(fit_mean) {
                tan_2omega_tau = (f2(i, j).imag() - 2 * fw(i, j).imag() * fw(i, j).real()) /
                    (f2(i, j).real() - (fw(i, j).real() * fw(i, j).real() - fw(i, j).imag() * fw(i, j).imag()));
            } else {
                tan_2omega_tau = f2(i, j).imag() / f2(i, j).real();
            }
            Scalar S2w = tan_2omega_tau / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar C2w = 1 / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar Cw = SQRT_HALF * std::sqrt(1 + C2w);
            Scalar Sw = SQRT_HALF * (S2w >= 0 ? 1 : -1) * std::sqrt(1 - C2w);

            Scalar YC = f1(i, j).real() * Cw + f1(i, j).imag() * Sw;
            Scalar YS = f1(i, j).imag() * Cw - f1(i, j).real() * Sw;
            Scalar CC = (1 + f2(i, j).real() * C2w + f2(i, j).imag() * S2w) / 2;
            Scalar SS = (1 - f2(i, j).real() * C2w - f2(i, j).imag() * S2w) / 2;

            if(fit_mean) {
                Scalar CC_fac = fw(i, j).real() * Cw + fw(i, j).imag() * Sw;
                Scalar SS_fac = fw(i, j).imag() * Cw - fw(i, j).real() * Sw;
                CC -= CC_fac * CC_fac;
                SS -= SS_fac * SS_fac;
            }

            power(i, j) = YC * YC / CC + YS * YS / SS;
            
            switch(norm_kind) {
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

NB_MODULE(cpu_helpers, m) {
    // We're using noconvert() here to ensure the input arrays are not copied
    
    m.def("process_finufft_inputs", &process_finufft_inputs<double>,
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

    m.def("process_finufft_inputs", &process_finufft_inputs<float>,
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

    m.def("process_finufft_outputs", &process_finufft_outputs<double>,
        "power"_a.noconvert(),
        "f1"_a.noconvert(),
        "fw"_a.noconvert(),
        "f2"_a.noconvert(),
        "norm_YY"_a.noconvert(),
        "norm_kind"_a,
        "fit_mean"_a,
        "nthreads"_a
        );

    m.def("process_finufft_outputs", &process_finufft_outputs<float>,
        "power"_a.noconvert(),
        "f1"_a.noconvert(),
        "fw"_a.noconvert(),
        "f2"_a.noconvert(),
        "norm_YY"_a.noconvert(),
        "norm_kind"_a,
        "fit_mean"_a,
        "nthreads"_a
        );

    nb::enum_<NormKind>(m, "NormKind")
        .value("Standard", NormKind::Standard)
        .value("Model", NormKind::Model)
        .value("Log", NormKind::Log)
        .value("PSD", NormKind::PSD);
}
