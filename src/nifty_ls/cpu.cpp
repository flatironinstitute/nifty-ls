#include <complex>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

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
    nifty_arr_1d<Scalar> norm_,
    nifty_arr_1d<const Scalar> t_,
    nifty_arr_2d<const Scalar> y_,
    nifty_arr_2d<const Scalar> dy_,
    const Scalar fmin,
    const Scalar df,
    const size_t Nf,
    const bool psd_norm
) {
    auto t1 = t1_.view();
    auto t2 = t2_.view();
    auto yw = yw_.view();
    auto w  = w_ .view();
    auto norm = norm_.view();
    auto t  = t_ .view();  // read-only
    auto y  = y_ .view();  // read-only
    auto dy = dy_.view();  // read-only

    size_t Nseq = y.shape(0);
    size_t N = y.shape(1);
    size_t Nshift = Nf / 2;

    // w = dy**-2.
    std::vector<double> sum(Nseq, 0.);  // use double for stability
    for (size_t i = 0; i < Nseq; ++i) {
        for (size_t j = 0; j < N; ++j) {
            w(i, j) = 1 / (dy(i, j) * dy(i, j));
            sum[i] += w(i, j).real();
        }
    }

    // norm = (w * y**2).sum(axis=-1, keepdims=True)
    // TODO: currently we're storing w as complex
    // but it's real at this point in the code, before the phase shift
    for (size_t i = 0; i < Nbatch; ++i) {
        if (psd_norm) {
            norm(i) = sum[i];
        }
        else {
            norm(i) = 0;
        }
        for (size_t j = 0; j < N; ++j) {
            w(i, j).real( w(i, j).real() / sum[i] );  // w /= sum
            if (!psd_norm) {
                norm(i) += w(i, j).real() * y(i, j) * y(i, j);
            }
        }
    }

    // TODO: if fmin/df is an integer, we may be able to do
    //   t = ((df * t) % 1) * 2 * np.pi
    // Could help a lot with range-reduction performance.
    const std::complex<Scalar> phase_shift = std::complex<Scalar>(0, Nshift + fmin/df);
    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);
    for(size_t j = 0; j < N; ++j) {
        t1(j) = TWO_PI * df * t(j);
        t2(j) = 2 * t1(j);

        for(size_t i = 0; i < Nbatch; ++i) {
            yw(i, j) = y(i, j) * w(i, j) * std::exp(phase_shift * t1(j));
            w(i, j) *= std::exp(phase_shift * t2(j));
        }

        t1(j) = std::fmod(t1(j), TWO_PI);
        t2(j) = std::fmod(t2(j), TWO_PI);
    }
}


template <typename Scalar>
void process_finufft_outputs(
    nifty_arr_2d<Scalar> power_,
    nifty_arr_2d<const std::complex<Scalar>> f1_,
    nifty_arr_2d<const std::complex<Scalar>> f2_,
    nifty_arr_1d<const Scalar> norm_YY_,
    const NormKind norm_kind
){
    auto power = power_.view();
    auto f1 = f1_.view();  // read-only
    auto f2 = f2_.view();  // read-only
    auto norm_YY = norm_YY_.view();  // read-only

    size_t Nbatch = f1.shape(0);
    size_t N = f1.shape(1);

    const Scalar SQRT_HALF = std::sqrt(0.5);

    for(size_t i = 0; i < Nbatch; ++i) {
        for(size_t j = 0; j < N; ++j) {
            Scalar tan_2omega_tau = f2(i, j).imag() / f2(i, j).real();
            Scalar S2w = tan_2omega_tau / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar C2w = 1 / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
            Scalar Cw = SQRT_HALF * std::sqrt(1 + C2w);
            Scalar Sw = SQRT_HALF * (S2w >= 0 ? 1 : -1) * std::sqrt(1 - C2w);

            Scalar YC = f1(i, j).real() * Cw + f1(i, j).imag() * Sw;
            Scalar YS = f1(i, j).imag() * Cw - f1(i, j).real() * Sw;
            Scalar CC = (1 + f2(i, j).real() * C2w + f2(i, j).imag() * S2w) / 2;
            Scalar SS = (1 - f2(i, j).real() * C2w - f2(i, j).imag() * S2w) / 2;

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
                    // instead of (w * y**2).sum()
                    power(i, j) *= 0.5 * norm_YY(i);
                    break;
            }
        }
    }
}

NB_MODULE(cpu, m) {
    m.def("process_finufft_inputs", &process_finufft_inputs<double>,
        "t1"_a.noconvert(),
        "t2"_a.noconvert(),
        "yw"_a.noconvert(),
        "w"_a.noconvert(),
        "norm"_a.noconvert(),
        "t"_a.noconvert(),
        "y"_a.noconvert(),
        "dy"_a.noconvert(),
        "fmin"_a,
        "df"_a,
        "Nf"_a,
        "center_data"_a,
        "fit_mean"_a,
        "psd_normalization"_a
        );
    m.def("process_finufft_inputs", &process_finufft_inputs<float>,
        "t1"_a.noconvert(),
        "t2"_a.noconvert(),
        "yw"_a.noconvert(),
        "w"_a.noconvert(),
        "norm"_a.noconvert(),
        "t"_a.noconvert(),
        "y"_a.noconvert(),
        "dy"_a.noconvert(),
        "fmin"_a,
        "df"_a,
        "Nf"_a,
        "center_data"_a,
        "fit_mean"_a,
        "psd_normalization"_a
        );
    m.def("process_finufft_outputs", &process_finufft_outputs<double>,
        "power"_a.noconvert(),
        "f1"_a.noconvert(),
        "f2"_a.noconvert(),
        "norm_YY"_a.noconvert(),
        "norm_kind"_a
        );
    m.def("process_finufft_outputs", &process_finufft_outputs<float>,
        "power"_a.noconvert(),
        "f1"_a.noconvert(),
        "f2"_a.noconvert(),
        "norm_YY"_a.noconvert(),
        "norm_kind"_a
        );

    nb::enum_<NormKind>(m, "NormKind")
        .value("Standard", NormKind::Standard)
        .value("Model", NormKind::Model)
        .value("Log", NormKind::Log)
        .value("PSD", NormKind::PSD);
}
