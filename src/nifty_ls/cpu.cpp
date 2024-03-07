#include <complex>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

namespace nb = nanobind;
using namespace nb::literals;

const double PI = 3.14159265358979323846;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using Complex = std::complex<Scalar>;

template <typename Scalar>
Scalar process_finufft_inputs(
    nifty_arr_1d<Scalar> t1_,
    nifty_arr_1d<Scalar> t2_,
    nifty_arr_1d<Complex<Scalar>> yw_,
    nifty_arr_1d<Complex<Scalar>> w_,
    nifty_arr_1d<const Scalar> t_,
    nifty_arr_1d<const Scalar> y_,
    nifty_arr_1d<const Scalar> dy_,
    const Scalar fmin,
    const Scalar df,
    const size_t Nf
) {
    auto t1 = t1_.view();
    auto t2 = t2_.view();
    auto yw = yw_.view();
    auto w  = w_ .view();
    auto t  = t_ .view();  // read-only
    auto y  = y_ .view();  // read-only
    auto dy = dy_.view();  // read-only

    size_t N = t.shape(0);
    size_t Nshift = Nf / 2;

    // w = dy**-2.
    double sum = 0.0;  // use double for stability
    for(size_t i = 0; i < N; ++i) {
        w(i) = 1 / (dy(i) * dy(i));
        sum += w(i).real();
    }
    // w /= w.sum()
    for(size_t i = 0; i < N; ++i) {
        w(i) /= sum;
    }

    // norm = np.dot(w, y ** 2)
    double norm = 0.0;
    for(size_t i = 0; i < N; ++i) {
        // TODO: currently we're storing w as complex
        // but it's real at this point in the code, before the phase shift
        norm += w(i).real() * y(i) * y(i);
    }

    // TODO: if fmin/df is an integer, we may be able to do
    //   t = ((df * t) % 1) * 2 * np.pi
    // Could help a lot with range-reduction performance.
    const std::complex<Scalar> phase_shift = std::complex<Scalar>(0, Nshift + fmin/df);
    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);
    for(size_t i = 0; i < N; ++i) {
        t1(i) = (TWO_PI * df * t(i));
        t2(i) = 2 * t1(i);

        yw(i) = y(i) * w(i) * std::exp(phase_shift * t1(i));
        w(i) *= std::exp(phase_shift * t2(i));

        t1(i) = std::fmod(t1(i), TWO_PI);
        t2(i) = std::fmod(t2(i), TWO_PI);
    }

    return static_cast<Scalar>(norm);
}


template <typename Scalar>
void process_finufft_outputs(
    nifty_arr_1d<Scalar> power_,
    nifty_arr_1d<const std::complex<Scalar>> f1_,
    nifty_arr_1d<const std::complex<Scalar>> f2_,
    const Scalar norm
){
    auto power = power_.view();
    auto f1 = f1_.view();  // read-only
    auto f2 = f2_.view();  // read-only
    size_t N = f1.shape(0);

    const Scalar SQRT_HALF = std::sqrt(0.5);
    const Scalar invnorm = 1 / norm;

    for(size_t i = 0; i < N; ++i) {
        // TODO: is it possible we're supposed to be doing some of these with complex operations?
        Scalar tan_2omega_tau = f2(i).imag() / f2(i).real();
        Scalar S2w = tan_2omega_tau / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
        Scalar C2w = 1 / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
        Scalar Cw = SQRT_HALF * std::sqrt(1 + C2w);
        Scalar Sw = SQRT_HALF * (S2w >= 0 ? 1 : -1) * std::sqrt(1 - C2w);

        Scalar YC = f1(i).real() * Cw + f1(i).imag() * Sw;
        Scalar YS = f1(i).imag() * Cw - f1(i).real() * Sw;
        Scalar CC = (1 + f2(i).real() * C2w + f2(i).imag() * S2w) / 2;
        Scalar SS = (1 - f2(i).real() * C2w - f2(i).imag() * S2w) / 2;

        power(i) = (YC * YC / CC + YS * YS / SS) * invnorm;
    }
}

NB_MODULE(cpu, m) {
    m.def("process_finufft_inputs", &process_finufft_inputs<double>,
        "t1"_a.noconvert(), "t2"_a.noconvert(), "yw"_a.noconvert(), "w"_a.noconvert(),
        "t"_a.noconvert(), "y"_a.noconvert(), "dy"_a.noconvert(), "fmin"_a, "df"_a, "Nf"_a);
    m.def("process_finufft_inputs", &process_finufft_inputs<float>,
        "t1"_a.noconvert(), "t2"_a.noconvert(), "yw"_a.noconvert(), "w"_a.noconvert(),
        "t"_a.noconvert(), "y"_a.noconvert(), "dy"_a.noconvert(), "fmin"_a, "df"_a, "Nf"_a);
    m.def("process_finufft_outputs", &process_finufft_outputs<double>,
        "power"_a.noconvert(), "f1"_a.noconvert(), "f2"_a.noconvert(), "norm"_a);
    m.def("process_finufft_outputs", &process_finufft_outputs<float>,
        "power"_a.noconvert(), "f1"_a.noconvert(), "f2"_a.noconvert(), "norm"_a);
}
