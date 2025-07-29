/* This module contains C++ helper routines for processing finufft_chi2 method
 * inputs and outputs. Its main purpose is to enable "kernel fusion",
 * i.e. do as much array processing as possible element-wise, instead
 * of array-wise as occurs in Numpy.
 */
#include <iostream>

#include <algorithm>
#include <complex>
#include <vector>

#include <cmath>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "cpu_helpers.hpp"
using cpu_helpers::NormKind;
using cpu_helpers::TermType;

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
    auto t1   = t1_.view();
    auto yw   = yw_.view();
    auto w    = w_.view();
    auto w2s  = w2s_.view();
    auto norm = norm_.view();
    auto yws  = yws_.view();
    auto Sw   = Sw_.view();  // shape (Nbatch, nSW, Nf)
    auto Cw   = Cw_.view();
    auto Syw  = Syw_.view();
    auto Cyw  = Cyw_.view();
    auto t    = t_.view();   // read-only
    auto y    = y_.view();   // read-only
    auto dy   = dy_.view();  // read-only
    size_t Nf = Sw.shape(2);

    size_t Nbatch = y.shape(0);
    size_t N      = y.shape(1);

    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);

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
    (void) nthreads;
#endif

// Compute and store t1
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (size_t j = 0; j < N; ++j) { t1(j) = TWO_PI * df * t(j); }

    // Process each batch serially, but parallelize inner loops
    for (size_t i = 0; i < Nbatch; ++i) {
        Scalar sum_w    = Scalar(0);
        Scalar yoff     = Scalar(0);
        Scalar sum_norm = Scalar(0);
        Scalar sum_yw2  = Scalar(0);

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
        {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(+ : sum_w, yoff)
#endif
            // 1. compute sum_w, yoff and fill w2s
            for (size_t j = 0; j < N; ++j) {
                Scalar wt = Scalar(1) / (dy(i, j) * dy(i, j));
                sum_w += wt;
                yoff += wt * y(i, j);
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
                w2s(i, 0) = sum_w;

                if (center_data || fit_mean) {
                    yoff /= sum_w;
                } else {
                    yoff = Scalar(0);
                }
            }

#ifdef _OPENMP
#pragma omp for schedule(static) reduction(+ : sum_norm, sum_yw2)
#endif
            // 2. compute norm, yws, and fill yw, w
            for (size_t m = 0; m < N; ++m) {
                Scalar wt = Scalar(1) / (dy(i, m) * dy(i, m));
                Scalar ym = y(i, m) - yoff;
                sum_norm += wt * (ym * ym);
                sum_yw2 += ym * wt;

                yw(i, m) = Complex<Scalar>(ym * wt, Scalar(0));
                w(i, m)  = Complex<Scalar>(wt, Scalar(0));
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
                norm(i, 0) = sum_norm;
                if (center_data || fit_mean) {
                    yws(i, 0) = Scalar(
                       0
                    );  // Mathematically, fit_mean or center_data will set yws to 0
                } else {
                    yws(i, 0) = sum_yw2;
                }
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            // 3. initialize trig matrix
            for (size_t f = 0; f < Nf; ++f) {
                Sw(i, 0, f)  = Scalar(0);
                Syw(i, 0, f) = Scalar(0);
                Cw(i, 0, f)  = w2s(i, 0);
                Cyw(i, 0, f) = yws(i, 0);
            }
        }
    }
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
#ifndef _OPENMP
    (void) nthreads;
#endif

    auto t1             = t1_.view();         // input length-N array
    auto yw_w           = yw_w_.view();       // input with 2*Nbatch × N size
    auto tn             = tn_out.view();      // output
    auto yw_s           = yw_w_s_out.view();  // output same shape as yw_w_
    const size_t N      = t1.shape(0);
    const size_t nTrans = yw_w.shape(0);

    Scalar factor = Scalar(Nf / 2) + fmin / df;  // shift factor

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        // tn = time_shift * t1
        for (size_t j = 0; j < N; ++j) { tn(j) = Scalar(time_shift) * t1(j); }

#ifdef _OPENMP
        size_t chunk_size = std::max(size_t(8), N / nthreads);
#pragma omp for schedule(static, chunk_size)
#endif
        // Do phase shift: phase_shift = np.exp(1j * ((Nf // 2) + fmin / df) * tn)
        for (size_t j = 0; j < N; ++j) {
            Complex<Scalar> phase = std::exp(Complex<Scalar>(0, factor * tn(j)));
            for (size_t b = 0; b < nTrans; ++b) { yw_s(b, j) = yw_w(b, j) * phase; }
        }
    }
}

// Solver for small matrices using LU decomposition with partial pivoting
// Using single thread
template <typename Scalar>
void small_matrixs_solver(std::vector<Scalar> &A, std::vector<Scalar> &B, size_t n) {
    // Tolerance for detecting singularity
    const Scalar tol = static_cast<Scalar>(1e-9);

    if (n == 2)  // min of n
    {
        // Solve a 2x2 system directly using the determinant
        Scalar det = A[0] * A[3] - A[1] * A[2];  // a11*a22 - a12*a21
        if (std::abs(det) < tol) { throw std::runtime_error("Matrix is singular"); }
        Scalar inv_det = static_cast<Scalar>(1.0) / det;
        Scalar b0      = B[0];
        Scalar b1      = B[1];
        // Solution using inverse matrix
        B[0] = (A[3] * b0 - A[2] * b1) * inv_det;   // (a22*b1 - a21*b2) / det
        B[1] = (-A[1] * b0 + A[0] * b1) * inv_det;  // (-a12*b1 + a11*b2) / det
    } else {
        // LU decomposition with partial pivoting for n > 2
        for (size_t k = 0; k < n; ++k) {
            // Find pivot
            size_t pivot   = k;
            Scalar max_val = std::abs(A[k * n + k]);
            for (size_t i = k + 1; i < n; ++i) {
                Scalar val = std::abs(A[i * n + k]);
                if (val > max_val) {
                    max_val = val;
                    pivot   = i;
                }
            }
            if (max_val < tol) { throw std::runtime_error("Matrix is singular"); }
            if (pivot != k) {
                // Swap rows k and pivot in A
                for (size_t j = 0; j < n; ++j) {
                    std::swap(A[k * n + j], A[pivot * n + j]);
                }
                // Apply the same swap to B
                std::swap(B[k], B[pivot]);
            }
            // Gaussian elimination to form L and U
            for (size_t j = k + 1; j < n; ++j) {
                Scalar m     = A[j * n + k] / A[k * n + k];  // Multiplier
                A[j * n + k] = m;                            // Store L (below diagonal)
                for (size_t l = k + 1; l < n; ++l) {
                    A[j * n + l] -= m * A[k * n + l];  // Update U
                }
            }
        }

        // Forward substitution: Solve L * Y = B
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) { B[i] -= A[i * n + j] * B[j]; }
            // L has 1s on diagonal, so no division needed
        }

        // Backward substitution: Solve U * X = Y
        for (size_t i = n; i-- > 0;) {
            for (size_t j = i + 1; j < n; ++j) { B[i] -= A[i * n + j] * B[j]; }
            B[i] /= A[i * n + i];  // Divide by U's diagonal element
        }
    }
}

// Single thread dot product for small vector
template <typename Scalar>
Scalar small_matrixs_dot(int n, const Scalar *x, const Scalar *y) {
    Scalar result = 0.0;
    for (int i = 0; i < n; ++i) { result += x[i] * y[i]; }
    return result;
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
    auto power              = power_.view();       // output
    const size_t order_size = order_types.size();  // input

    auto Sw   = Sw_.view();    // input (Nbatch, nSW, Nf)
    auto Cw   = Cw_.view();    // input
    auto Syw  = Syw_.view();   // input
    auto Cyw  = Cyw_.view();   // input
    auto norm = norm_.view();  // input

    const size_t Nbatch = Sw.shape(0);
    const size_t nSW    = Sw.shape(1);
    const size_t Nf     = Sw.shape(2);

#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
#else
    (void) nthreads;  // suppress unused variable warning
#endif

// Process each batch and frequency
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        // Allocate arrays for each thread
        std::vector<Scalar> XTy(order_size);
        std::vector<Scalar> XTX(order_size * order_size);  // Flat matrix for XTX
        std::vector<Scalar> bvec(order_size);
        std::vector<Scalar> sw_local(nSW);
        std::vector<Scalar> cw_local(nSW);
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (size_t b = 0; b < Nbatch; ++b) {
            for (size_t f = 0; f < Nf; ++f) {
                Scalar norm_val = norm(b, 0);
                // Prefetch data into local arrays
                for (size_t i = 0; i < nSW; ++i) {
                    sw_local[i] = Sw(b, i, f);
                    cw_local[i] = Cw(b, i, f);
                }

                // Fill XTy with proper indexing for new array layout
                for (size_t i = 0; i < order_size; ++i) {
                    TermType t = order_types[i];
                    size_t m   = order_indices[i];
                    XTy[i]     = (t == TermType::Sine) ? Syw(b, m, f) : Cyw(b, m, f);
                }

                // Fill XTX efficiently using local arrays
                for (size_t i = 0; i < order_size; ++i) {
                    TermType ti = order_types[i];
                    size_t m    = order_indices[i];
                    for (size_t j = 0; j < order_size; ++j) {
                        TermType tj = order_types[j];
                        size_t n    = order_indices[j];

                        size_t d = (m > n) ? (m - n) : (n - m);
                        size_t s = m + n;

                        if (ti == TermType::Sine && tj == TermType::Sine) {
                            XTX[j * order_size + i] =
                               Scalar(0.5) * (cw_local[d] - cw_local[s]);
                        } else if (ti == TermType::Cosine && tj == TermType::Cosine) {
                            XTX[j * order_size + i] =
                               Scalar(0.5) * (cw_local[d] + cw_local[s]);
                        } else if (ti == TermType::Sine && tj == TermType::Cosine) {
                            int sign = (m > n ? 1 : (m < n ? -1 : 0));
                            XTX[j * order_size + i] =
                               Scalar(0.5) * (sign * sw_local[d] + sw_local[s]);
                        } else {
                            int sign = (n > m ? 1 : (n < m ? -1 : 0));
                            XTX[j * order_size + i] =
                               Scalar(0.5) * (sign * sw_local[d] + sw_local[s]);
                        }
                    }
                }

                // Copy XTy to preserve it for dot product later
                std::copy(XTy.begin(), XTy.end(), bvec.begin());

                // Custom Solver
                size_t n = order_size;
                try {
                    small_matrixs_solver(XTX, bvec, n);
                } catch (const std::exception &e) {
                    throw std::runtime_error(
                       "Custom solver failed: " + std::string(e.what())
                    );
                }

                // Dot product (XTy, bvec)
                Scalar pw = small_matrixs_dot(n, bvec.data(), XTy.data());

                // Apply normalization
                switch (norm_kind) {
                    case NormKind::Standard:
                        pw /= norm_val;
                        break;
                    case NormKind::Model:
                        pw /= (norm_val - pw);
                        break;
                    case NormKind::Log:
                        pw = -std::log(1 - pw / norm_val);
                        break;
                    case NormKind::PSD:
                        pw *= Scalar(0.5);
                        break;
                }
                power(b, f) = pw;  // write to output array
            }
        }
    }
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
