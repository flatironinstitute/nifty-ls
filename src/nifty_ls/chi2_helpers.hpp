/* This module contains C++ helper routines for processing finufft_chi2 method
 * inputs and outputs.
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

#include "utils_helpers.hpp"
using utils_helpers::NormKind;
using utils_helpers::TermType;

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
using Complex = std::complex<Scalar>;

template <typename Scalar>
void process_chi2_inputs_raw(
   Scalar *t1,           // (N)
   Complex<Scalar> *yw,  // (Nbatch, N)
   Complex<Scalar> *w,   // (Nbatch, N)
   Scalar *w2s,          // (Nbatch, 1)
   Scalar *norm,         // (Nbatch, 1)
   Scalar *yws,          // (Nbatch, 1)
   Scalar *Sw,           // (Nbatch, nSW, Nf)
   Scalar *Cw,           // (Nbatch, nSW, Nf)
   Scalar *Syw,          // (Nbatch, nSY, Nf)
   Scalar *Cyw,          // (Nbatch, nSY, Nf)
   const Scalar *t,      // input, (N)
   const Scalar *y,      // input, (Nbatch, N)
   const Scalar *dy,     // input, (Nbatch, N)
   const size_t Nbatch,
   const size_t N,
   const size_t Nf,
   const size_t nSW,
   const size_t nSY,
   const Scalar df,
   const bool center_data,
   const bool fit_mean,
   int nthreads
   // TODO: add if to control if OMP
) {
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
    for (size_t j = 0; j < N; ++j) { t1[j] = TWO_PI * df * t[j]; }

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
                Scalar wt = Scalar(1) / (dy[i * N + j] * dy[i * N + j]);
                sum_w += wt;
                yoff += wt * y[i * N + j];
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
                w2s[i] = sum_w;

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
                Scalar wt = Scalar(1) / (dy[i * N + m] * dy[i * N + m]);
                Scalar ym = y[i * N + m] - yoff;
                sum_norm += wt * (ym * ym);
                sum_yw2 += ym * wt;

                yw[i * N + m] = Complex<Scalar>(ym * wt, Scalar(0));
                w[i * N + m]  = Complex<Scalar>(wt, Scalar(0));
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
                norm[i] = sum_norm;
                // Mathematically, fit_mean or center_data will set yws to 0
                if (center_data || fit_mean) {
                    yws[i] = Scalar(0);
                } else {
                    yws[i] = sum_yw2;
                }
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            // 3. initialize trig matrix
            for (size_t f = 0; f < Nf; ++f) {
                Sw[i * nSW * Nf + 0 * Nf + f]  = Scalar(0);
                Syw[i * nSY * Nf + 0 * Nf + f] = Scalar(0);
                Cw[i * nSW * Nf + 0 * Nf + f]  = w2s[i];
                Cyw[i * nSY * Nf + 0 * Nf + f] = yws[i];
            }
        }
    }
}

template <typename Scalar>
void compute_t_raw(
   const Scalar *t1,             // input, (N)
   const Complex<Scalar> *yw_w,  // input, (nTrans, N)
   const size_t time_shift,
   const size_t N,
   const size_t nTrans,
   const Scalar factor,
   Scalar *tn_out,               // (N)
   Complex<Scalar> *yw_w_s_out,  // (nTrans, N)
   int nthreads
) {
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t j = 0; j < N; ++j) { tn_out[j] = Scalar(time_shift) * t1[j]; }

#ifdef _OPENMP
        size_t chunk_size = std::max(size_t(8), N / nthreads);
#pragma omp for schedule(static, chunk_size)
#endif
        // Do phase shift: phase_shift = exp(1j * factor * tn)

        for (size_t j = 0; j < N; ++j) {
            Complex<Scalar> phase = std::exp(Complex<Scalar>(0, factor * tn_out[j]));
            for (size_t b = 0; b < nTrans; ++b) {
                yw_w_s_out[b * N + j] = yw_w[b * N + j] * phase;
            }
        }
    }
}

// Solver for small matrices using LU decomposition with partial pivoting
// Using single thread
template <typename Scalar>
void small_matrixs_solver(
   std::vector<Scalar> &A, std::vector<Scalar> &B, const size_t n
) {
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
Scalar small_matrixs_dot(const int n, const Scalar *x, const Scalar *y) {
    Scalar result = 0.0;
    for (int i = 0; i < n; ++i) { result += x[i] * y[i]; }
    return result;
}

template <typename Scalar>
void process_chi2_outputs_raw(
   Scalar *power,       // (Nbatch, Nf)
   const Scalar *Sw,    // input, (Nbatch, nSW, Nf)
   const Scalar *Cw,    // input, (Nbatch, nSW, Nf)
   const Scalar *Syw,   // input, (Nbatch, nSY, Nf)
   const Scalar *Cyw,   // input, (Nbatch, nSY, Nf)
   const Scalar *norm,  // input, (Nbatch, 1)
   const std::vector<TermType> &order_types,
   const std::vector<size_t> &order_indices,
   const size_t Nbatch,
   const size_t nSW,
   const size_t nSY,
   const size_t Nf,
   const NormKind norm_kind,
   int nthreads
) {
    const size_t order_size = order_types.size();

#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
#else
    (void) nthreads;
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        std::vector<Scalar> XTy(order_size);
        std::vector<Scalar> XTX(order_size * order_size);
        std::vector<Scalar> bvec(order_size);
        std::vector<Scalar> sw_local(nSW);
        std::vector<Scalar> cw_local(nSW);
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (size_t b = 0; b < Nbatch; ++b) {
            for (size_t f = 0; f < Nf; ++f) {
                Scalar norm_val = norm[b];
                for (size_t i = 0; i < nSW; ++i) {
                    sw_local[i] = Sw[b * nSW * Nf + i * Nf + f];
                    cw_local[i] = Cw[b * nSW * Nf + i * Nf + f];
                }
                for (size_t i = 0; i < order_size; ++i) {
                    TermType t = order_types[i];
                    size_t m   = order_indices[i];
                    XTy[i] = (t == TermType::Sine) ? Syw[b * nSY * Nf + m * Nf + f] :
                                                     Cyw[b * nSY * Nf + m * Nf + f];
                }
                for (size_t i = 0; i < order_size; ++i) {
                    TermType ti = order_types[i];
                    size_t m    = order_indices[i];
                    for (size_t j = 0; j < order_size; ++j) {
                        TermType tj = order_types[j];
                        size_t n    = order_indices[j];
                        size_t d    = (m > n) ? (m - n) : (n - m);
                        size_t s    = m + n;
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
                size_t n = order_size;
                try {  // Using Custom Solver
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
                power[b * Nf + f] = pw;
            }
        }
    }
}
