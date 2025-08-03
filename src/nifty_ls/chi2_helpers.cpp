/* This module is a nanobind wrapper for processing finufft_chi2 method
 * inputs and outputs. Its main purpose is to enable "kernel fusion",
 * i.e. do as much array processing as possible element-wise, instead
 * of array-wise as occurs in Numpy.
 */

#include <algorithm>
#include <complex>
#include <vector>

#include <cmath>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "chi2_helpers.hpp"
#include "utils_helpers.hpp"
using utils_helpers::NormKind;
using utils_helpers::TermType;

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

template <typename Scalar>
void process_chi2_inputs(
   nifty_arr_1d<Scalar> t1_,
   nifty_arr_2d<Complex<Scalar>> yw_,
   nifty_arr_2d<Complex<Scalar>> w_,
   nifty_arr_2d<Scalar> w2s_,
   nifty_arr_2d<Scalar> norm_,
   nifty_arr_2d<Scalar> yws_,
   nifty_arr_3d<Scalar> Sw_,
   nifty_arr_3d<Scalar> Cw_,
   nifty_arr_3d<Scalar> Syw_,
   nifty_arr_3d<Scalar> Cyw_,
   nifty_arr_1d<const Scalar> t_,
   nifty_arr_2d<const Scalar> y_,
   nifty_arr_2d<const Scalar> dy_,
   const Scalar df,
   const bool center_data,
   const bool fit_mean,
   int nthreads
) {
    // Extract pointers and shapes
    Scalar *t1          = t1_.data();
    Complex<Scalar> *yw = yw_.data();
    Complex<Scalar> *w  = w_.data();
    Scalar *w2s         = w2s_.data();
    Scalar *norm        = norm_.data();
    Scalar *yws         = yws_.data();
    Scalar *Sw          = Sw_.data();
    Scalar *Cw          = Cw_.data();
    Scalar *Syw         = Syw_.data();
    Scalar *Cyw         = Cyw_.data();
    const Scalar *t     = t_.data();
    const Scalar *y     = y_.data();
    const Scalar *dy    = dy_.data();

    size_t Nbatch = y_.shape(0);
    size_t N      = y_.shape(1);
    size_t Nf     = Sw_.shape(2);
    size_t nSW    = Sw_.shape(1);
    size_t nSY    = Syw_.shape(1);

    process_chi2_inputs_raw<Scalar>(
       t1,
       yw,
       w,
       w2s,
       norm,
       yws,
       Sw,
       Cw,
       Syw,
       Cyw,
       t,
       y,
       dy,
       Nbatch,
       N,
       Nf,
       nSW,
       nSY,
       df,
       center_data,
       fit_mean,
       nthreads
    );
}

template <typename Scalar>
void compute_t(
   nifty_arr_1d<const Scalar> &t1_,
   nifty_arr_2d<const Complex<Scalar>> &yw_w_,
   const size_t time_shift,
   const Scalar fmin,
   const Scalar df,
   const size_t Nf,
   nifty_arr_1d<Scalar> &tn_out,
   nifty_arr_2d<Complex<Scalar>> &yw_w_s_out,
   int nthreads
) {
    const Scalar *t1            = t1_.data();
    const Complex<Scalar> *yw_w = yw_w_.data();
    Scalar *tn                  = tn_out.data();
    Complex<Scalar> *yw_w_s     = yw_w_s_out.data();

    const size_t N      = t1_.shape(0);
    const size_t nTrans = yw_w_.shape(0);

    const Scalar factor = Scalar(Nf / 2) + fmin / df;  // shift factor

    compute_t_raw<Scalar>(
       t1, yw_w, time_shift, N, nTrans, factor, tn, yw_w_s, nthreads
    );
}

template <typename Scalar>
void process_chi2_outputs(
   nifty_arr_2d<Scalar> power_,
   nifty_arr_3d<const Scalar> Sw_,
   nifty_arr_3d<const Scalar> Cw_,
   nifty_arr_3d<const Scalar> Syw_,
   nifty_arr_3d<const Scalar> Cyw_,
   nifty_arr_2d<const Scalar> norm_,
   const std::vector<TermType> &order_types,  // Sine or Cosine
   const std::vector<size_t> &order_indices,  // Nterms
   const NormKind norm_kind,
   int nthreads
) {
    Scalar *power      = power_.data();
    const Scalar *Sw   = Sw_.data();
    const Scalar *Cw   = Cw_.data();
    const Scalar *Syw  = Syw_.data();
    const Scalar *Cyw  = Cyw_.data();
    const Scalar *norm = norm_.data();

    size_t Nbatch = Sw_.shape(0);
    size_t nSW    = Sw_.shape(1);
    size_t nSY    = Syw_.shape(1);
    size_t Nf     = Sw_.shape(2);

    process_chi2_outputs_raw<Scalar>(
       power,
       Sw,
       Cw,
       Syw,
       Cyw,
       norm,
       order_types,
       order_indices,
       Nbatch,
       nSW,
       nSY,
       Nf,
       norm_kind,
       nthreads
    );
}

NB_MODULE(chi2_helpers, m) {
    // We're using noconvert() here to ensure the input arrays are not copied

    m.def(
       "process_chi2_inputs",
       &process_chi2_inputs<double>,
       "t1"_a.noconvert(),
       "yw"_a.noconvert(),
       "w"_a.noconvert(),
       "w2s"_a.noconvert(),
       "norm"_a.noconvert(),
       "yws"_a.noconvert(),
       "Sw"_a.noconvert(),
       "Cw"_a.noconvert(),
       "Syw"_a.noconvert(),
       "Cyw"_a.noconvert(),
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "df"_a,
       "center_data"_a,
       "fit_mean"_a,
       "nthreads"_a
    );

    m.def(
       "process_chi2_inputs",
       &process_chi2_inputs<float>,
       "t1"_a.noconvert(),
       "yw"_a.noconvert(),
       "w"_a.noconvert(),
       "w2s"_a.noconvert(),
       "norm"_a.noconvert(),
       "yws"_a.noconvert(),
       "Sw"_a.noconvert(),
       "Cw"_a.noconvert(),
       "Syw"_a.noconvert(),
       "Cyw"_a.noconvert(),
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "df"_a,
       "center_data"_a,
       "fit_mean"_a,
       "nthreads"_a
    );

    m.def(
       "compute_t",
       &compute_t<double>,
       "t1"_a.noconvert(),
       "yw_w"_a.noconvert(),
       "time_shift"_a,
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "tn_out"_a.noconvert(),
       "yw_w_s_out"_a.noconvert(),
       "nthreads"_a
    );

    m.def(
       "compute_t",
       &compute_t<float>,
       "t1"_a.noconvert(),
       "yw_w"_a.noconvert(),
       "time_shift"_a,
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "tn_out"_a.noconvert(),
       "yw_w_s_out"_a.noconvert(),
       "nthreads"_a
    );

    m.def(
       "process_chi2_outputs",
       &process_chi2_outputs<double>,
       "power"_a,
       "Sw"_a.noconvert(),
       "Cw"_a.noconvert(),
       "Syw"_a.noconvert(),
       "Cyw"_a.noconvert(),
       "norm"_a.noconvert(),
       "order_types"_a,
       "order_indices"_a,
       "norm_kind"_a,
       "nthreads"_a
    );

    m.def(
       "process_chi2_outputs",
       &process_chi2_outputs<float>,
       "power"_a,
       "Sw"_a.noconvert(),
       "Cw"_a.noconvert(),
       "Syw"_a.noconvert(),
       "Cyw"_a.noconvert(),
       "norm"_a.noconvert(),
       "order_types"_a,
       "order_indices"_a,
       "norm_kind"_a,
       "nthreads"_a
    );

    nb::enum_<TermType>(m, "TermType")
       .value("Sine", TermType::Sine)
       .value("Cosine", TermType::Cosine);
}
