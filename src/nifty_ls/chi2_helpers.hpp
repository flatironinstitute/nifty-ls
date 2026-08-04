/* This module contains C++ helper routines for processing finufft_chi2 method
 * inputs and outputs.
 */

#include <algorithm>
#include <complex>
#include <vector>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils_helpers.hpp"
using utils_helpers::NormKind;
using utils_helpers::TermType;

namespace nb = nanobind;
using namespace nb::literals;

template <typename Scalar>
using Complex = std::complex<Scalar>;

#if defined(__AVX512F__)
#define SOLVER_VEC_BYTES 64
#elif defined(__AVX__)
#define SOLVER_VEC_BYTES 32
#else
#define SOLVER_VEC_BYTES 16
#endif

template <typename Scalar>
struct solver_vec_traits {
    typedef Scalar type __attribute__((vector_size(SOLVER_VEC_BYTES)));
};

template <typename Scalar>
using SOLVER_VEC = typename solver_vec_traits<Scalar>::type;

template <typename Scalar>
constexpr size_t SOLVER_VEC_LEN = sizeof(SOLVER_VEC<Scalar>) / sizeof(Scalar);

template <typename Scalar>
void process_chi2_inputs_raw(
   Scalar *t1,
   Complex<Scalar> *yw,
   Complex<Scalar> *w,
   Scalar *w2s,
   Scalar *norm,
   Scalar *yws,
   Scalar *Sw,
   Scalar *Cw,
   Scalar *Syw,
   Scalar *Cyw,
   const nifty_arr_1d<const Scalar> &t,
   const nifty_arr_2d<const Scalar> &y,
   const nifty_arr_2d<const Scalar> &dy,
   const size_t Nbatch,
   const size_t N,
   const size_t Nf,
   const size_t nSW,
   const size_t nSY,
   const Scalar df,
   const bool center_data,
   const bool fit_mean,
   int nthreads
) {
    const Scalar TWO_PI = 2 * static_cast<Scalar>(PI);

#ifdef _OPENMP
    if (nthreads != 1) {
        if (nthreads < 1) { nthreads = omp_get_max_threads(); }
        if (nthreads > omp_get_max_threads()) {
            fprintf(
               stderr,
               "[nifty-ls finufft] Warning: nthreads (%d) > omp_get_max_threads() (%d). Performance may be suboptimal.\n",
               nthreads,
               omp_get_max_threads()
            );
        }
    }
#else
    (void) nthreads;
#endif

// Compute and store t1
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) if (nthreads > 1)
#endif
    for (size_t j = 0; j < N; ++j) { t1[j] = TWO_PI * df * t(j); }

    // Process each batch serially, but parallelize inner loops
    for (size_t i = 0; i < Nbatch; ++i) {
        Scalar sum_w    = Scalar(0);
        Scalar yoff     = Scalar(0);
        Scalar sum_norm = Scalar(0);
        Scalar sum_yw2  = Scalar(0);

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
#endif
        {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(+ : sum_w, yoff)
#endif
            // 1. compute sum_w, yoff and fill w2s
            for (size_t j = 0; j < N; ++j) {
                // If dy is None, use unit weights
                Scalar d  = dy(i, j);
                Scalar wt = Scalar(1) / (d * d);
                sum_w += wt;
                yoff += wt * y(i, j);
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
                Scalar d  = dy(i, m);
                Scalar wt = Scalar(1) / (d * d);
                Scalar ym = y(i, m) - yoff;
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
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
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

template <typename Scalar>
SOLVER_VEC<Scalar> solver_vec_splat(const Scalar value) {
    SOLVER_VEC<Scalar> result{};
    for (size_t lane = 0; lane < SOLVER_VEC_LEN<Scalar>; ++lane) {
        result[lane] = value;
    }
    return result;
}

template <typename Scalar>
SOLVER_VEC<Scalar> solver_vec_sqrt(const SOLVER_VEC<Scalar> value) {
    SOLVER_VEC<Scalar> result{};
    for (size_t lane = 0; lane < SOLVER_VEC_LEN<Scalar>; ++lane) {
        result[lane] = std::sqrt(value[lane]);
    }
    return result;
}

inline bool solver_is_nonfinite(const double value) {
    union {
        double f;
        uint64_t u;
    } bits = {value};
    return (bits.u & UINT64_C(0x7fffffffffffffff)) >= UINT64_C(0x7ff0000000000000);
}

inline bool solver_is_nonfinite(const float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (bits.u & UINT32_C(0x7fffffff)) >= UINT32_C(0x7f800000);
}

template <typename Scalar>
bool solver_condition_is_singular(const Scalar condition_bound) {
    const Scalar threshold = static_cast<Scalar>(1e11);
    return solver_is_nonfinite(condition_bound) || condition_bound < Scalar(0)
           || condition_bound > threshold;
}

template <typename Scalar>
bool solver_complex_denominator_is_singular(const Scalar real, const Scalar imag) {
    const Scalar denom = real * real + imag * imag;
    return solver_is_nonfinite(denom) || denom <= Scalar(0);
}

// Levinson-Durbin recursion for lane-batched complex Hermitian Toeplitz systems.
template <typename Scalar>
SOLVER_VEC<Scalar> small_toeplitz_solver(
   const std::vector<SOLVER_VEC<Scalar>> &Rr,
   const std::vector<SOLVER_VEC<Scalar>> &Ri,
   const std::vector<SOLVER_VEC<Scalar>> &Yr,
   const std::vector<SOLVER_VEC<Scalar>> &Yi,
   std::vector<SOLVER_VEC<Scalar>> &Xr,
   std::vector<SOLVER_VEC<Scalar>> &Xi,
   std::vector<SOLVER_VEC<Scalar>> &Ar,
   std::vector<SOLVER_VEC<Scalar>> &Ai,
   std::vector<SOLVER_VEC<Scalar>> &Apr,
   std::vector<SOLVER_VEC<Scalar>> &Api,
   const size_t n
) {
    const SOLVER_VEC<Scalar> zero = solver_vec_splat<Scalar>(Scalar(0));
    const SOLVER_VEC<Scalar> one  = solver_vec_splat<Scalar>(Scalar(1));

    SOLVER_VEC<Scalar> E               = Rr[0];
    SOLVER_VEC<Scalar> condition_bound = one;

    Xr[0] = Yr[0] / E;
    Xi[0] = Yi[0] / E;
    Ar[0] = one;
    Ai[0] = zero;

    for (size_t k = 1; k < n; ++k) {
        SOLVER_VEC<Scalar> lambda_r = zero;
        SOLVER_VEC<Scalar> lambda_i = zero;

        for (size_t i = 0; i < k; ++i) {
            const SOLVER_VEC<Scalar> rr = Rr[k - i];
            const SOLVER_VEC<Scalar> ri = Ri[k - i];
            const SOLVER_VEC<Scalar> ar = Ar[i];
            const SOLVER_VEC<Scalar> ai = Ai[i];
            lambda_r += rr * ar - ri * ai;
            lambda_i += rr * ai + ri * ar;
        }

        const SOLVER_VEC<Scalar> gamma_r = -lambda_r / E;
        const SOLVER_VEC<Scalar> gamma_i = -lambda_i / E;

        for (size_t i = 0; i < k; ++i) {
            Apr[i] = Ar[i];
            Api[i] = Ai[i];
        }

        Ar[k] = gamma_r * Apr[0] - gamma_i * (-Api[0]);
        Ai[k] = gamma_r * (-Api[0]) + gamma_i * Apr[0];

        for (size_t i = 1; i < k; ++i) {
            const SOLVER_VEC<Scalar> apr = Apr[k - i];
            const SOLVER_VEC<Scalar> api = -Api[k - i];
            Ar[i]                        = Apr[i] + gamma_r * apr - gamma_i * api;
            Ai[i]                        = Api[i] + gamma_r * api + gamma_i * apr;
        }

        const SOLVER_VEC<Scalar> abs2_gamma = gamma_r * gamma_r + gamma_i * gamma_i;
        const SOLVER_VEC<Scalar> abs_gamma  = solver_vec_sqrt<Scalar>(abs2_gamma);
        condition_bound *= (one + abs_gamma) / (one - abs_gamma);
        E *= one - abs2_gamma;

        SOLVER_VEC<Scalar> mu_r = Yr[k];
        SOLVER_VEC<Scalar> mu_i = Yi[k];

        for (size_t i = 0; i < k; ++i) {
            const SOLVER_VEC<Scalar> rr = Rr[k - i];
            const SOLVER_VEC<Scalar> ri = Ri[k - i];
            const SOLVER_VEC<Scalar> xr = Xr[i];
            const SOLVER_VEC<Scalar> xi = Xi[i];
            mu_r -= rr * xr - ri * xi;
            mu_i -= rr * xi + ri * xr;
        }

        const SOLVER_VEC<Scalar> nu_r = mu_r / E;
        const SOLVER_VEC<Scalar> nu_i = mu_i / E;

        Xr[k] = nu_r * Ar[0] - nu_i * (-Ai[0]);
        Xi[k] = nu_r * (-Ai[0]) + nu_i * Ar[0];

        for (size_t i = 0; i < k; ++i) {
            const SOLVER_VEC<Scalar> ar = Ar[k - i];
            const SOLVER_VEC<Scalar> ai = -Ai[k - i];
            Xr[i] += nu_r * ar - nu_i * ai;
            Xi[i] += nu_r * ai + nu_i * ar;
        }
    }

    return condition_bound;
}

template <typename Scalar>
SOLVER_VEC<Scalar> small_toeplitz_dot(
   const std::vector<SOLVER_VEC<Scalar>> &Yr,
   const std::vector<SOLVER_VEC<Scalar>> &Yi,
   const std::vector<SOLVER_VEC<Scalar>> &Xr,
   const std::vector<SOLVER_VEC<Scalar>> &Xi,
   const size_t n
) {
    SOLVER_VEC<Scalar> result = solver_vec_splat<Scalar>(Scalar(0));
    for (size_t i = 0; i < n; ++i) { result += Yr[i] * Xr[i] + Yi[i] * Xi[i]; }
    return result;
}

template <typename Scalar>
void process_chi2_outputs_raw(
   Scalar *power,
   const Scalar *Sw,
   const Scalar *Cw,
   const Scalar *Syw,
   const Scalar *Cyw,
   const Scalar *norm,
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
    const bool fit_mean =
       order_size > 0 && order_types[0] == TermType::Cosine && order_indices[0] == 0;
    const size_t order_offset = fit_mean ? 1 : 0;

    if (order_size < order_offset || ((order_size - order_offset) % 2) != 0) {
        throw nb::value_error(
           "[nifty-ls finufft] Error: Unsupported chi2 model order for Toeplitz solve."
        );
    }

    const size_t degree        = (order_size - order_offset) / 2;
    const size_t toeplitz_size = 2 * degree + 1;
    const size_t center        = degree;

    if (nSW < toeplitz_size || nSY < degree + 1) {
        throw nb::value_error(
           "[nifty-ls finufft] Error: Insufficient chi2 trigonometric sums for Toeplitz solve."
        );
    }

    for (size_t i = 0; i < degree; ++i) {
        const size_t sin_pos = order_offset + 2 * i;
        const size_t cos_pos = sin_pos + 1;
        const size_t h       = i + 1;
        if (
           order_types[sin_pos] != TermType::Sine
           || order_types[cos_pos] != TermType::Cosine || order_indices[sin_pos] != h
           || order_indices[cos_pos] != h
        ) {
            throw nb::value_error(
               "[nifty-ls finufft] Error: Unsupported chi2 model order for Toeplitz solve."
            );
        }
    }

    size_t singular_count       = 0;
    const size_t solver_vec_len = SOLVER_VEC_LEN<Scalar>;
    const size_t freq_blocks    = (Nf + solver_vec_len - 1) / solver_vec_len;
    const Scalar nan            = std::numeric_limits<Scalar>::quiet_NaN();

#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
#else
    (void) nthreads;
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)         \
   reduction(+ : singular_count) if (nthreads > 1)
#endif
    {
        using Vec = SOLVER_VEC<Scalar>;

        const Vec zero = solver_vec_splat<Scalar>(Scalar(0));
        const Vec one  = solver_vec_splat<Scalar>(Scalar(1));

        std::vector<Vec> Rr(toeplitz_size);
        std::vector<Vec> Ri(toeplitz_size);
        std::vector<Vec> Yr(toeplitz_size);
        std::vector<Vec> Yi(toeplitz_size);
        std::vector<Vec> Xr(toeplitz_size);
        std::vector<Vec> Xi(toeplitz_size);
        std::vector<Vec> Ar(toeplitz_size);
        std::vector<Vec> Ai(toeplitz_size);
        std::vector<Vec> Apr(toeplitz_size);
        std::vector<Vec> Api(toeplitz_size);
        std::vector<Vec> center_rhs_r(toeplitz_size);
        std::vector<Vec> center_rhs_i(toeplitz_size);
        std::vector<Vec> center_xr(toeplitz_size);
        std::vector<Vec> center_xi(toeplitz_size);

        for (size_t k = 0; k < toeplitz_size; ++k) {
            center_rhs_r[k] = zero;
            center_rhs_i[k] = zero;
        }
        center_rhs_r[center] = one;

#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (size_t b = 0; b < Nbatch; ++b) {
            for (size_t block = 0; block < freq_blocks; ++block) {
                const size_t base = block * solver_vec_len;

                for (size_t k = 0; k < toeplitz_size; ++k) {
                    const ptrdiff_t h =
                       static_cast<ptrdiff_t>(k) - static_cast<ptrdiff_t>(degree);

                    for (size_t lane = 0; lane < solver_vec_len; ++lane) {
                        size_t f = base + lane;
                        if (f >= Nf) { f = Nf - 1; }

                        Rr[k][lane] = Cw[b * nSW * Nf + k * Nf + f];
                        Ri[k][lane] = -Sw[b * nSW * Nf + k * Nf + f];

                        if (h == 0) {
                            Yr[k][lane] = fit_mean ? Cyw[b * nSY * Nf + f] : Scalar(0);
                            Yi[k][lane] = Scalar(0);
                        } else if (h > 0) {
                            const size_t ah = static_cast<size_t>(h);
                            Yr[k][lane]     = Cyw[b * nSY * Nf + ah * Nf + f];
                            Yi[k][lane]     = -Syw[b * nSY * Nf + ah * Nf + f];
                        } else {
                            const size_t ah = static_cast<size_t>(-h);
                            Yr[k][lane]     = Cyw[b * nSY * Nf + ah * Nf + f];
                            Yi[k][lane]     = Syw[b * nSY * Nf + ah * Nf + f];
                        }
                    }
                }

                Vec condition_bound = small_toeplitz_solver<Scalar>(
                   Rr, Ri, Yr, Yi, Xr, Xi, Ar, Ai, Apr, Api, toeplitz_size
                );
                Vec center_condition_bound = one;

                if (!fit_mean) {
                    // Remove the central harmonic with a Schur complement when
                    // the bias term is not part of the fitted model.
                    center_condition_bound = small_toeplitz_solver<Scalar>(
                       Rr,
                       Ri,
                       center_rhs_r,
                       center_rhs_i,
                       center_xr,
                       center_xi,
                       Ar,
                       Ai,
                       Apr,
                       Api,
                       toeplitz_size
                    );

                    const Vec denom = center_xr[center] * center_xr[center]
                                      + center_xi[center] * center_xi[center];
                    const Vec ratio_r =
                       (Xr[center] * center_xr[center] + Xi[center] * center_xi[center])
                       / denom;
                    const Vec ratio_i =
                       (Xi[center] * center_xr[center] - Xr[center] * center_xi[center])
                       / denom;

                    for (size_t k = 0; k < toeplitz_size; ++k) {
                        const Vec tr = center_xr[k] * ratio_r - center_xi[k] * ratio_i;
                        const Vec ti = center_xr[k] * ratio_i + center_xi[k] * ratio_r;
                        Xr[k] -= tr;
                        Xi[k] -= ti;
                    }
                }

                const Vec dot =
                   small_toeplitz_dot<Scalar>(Yr, Yi, Xr, Xi, toeplitz_size);

                for (size_t lane = 0; lane < solver_vec_len; ++lane) {
                    const size_t f = base + lane;
                    if (f >= Nf) { continue; }

                    bool is_singular =
                       solver_condition_is_singular(condition_bound[lane]);
                    if (!fit_mean) {
                        is_singular =
                           is_singular
                           || solver_condition_is_singular(center_condition_bound[lane])
                           || solver_complex_denominator_is_singular(
                              center_xr[center][lane], center_xi[center][lane]
                           );
                    }

                    if (is_singular) {
                        power[b * Nf + f] = nan;
                        ++singular_count;
                        continue;
                    }

                    Scalar pw       = dot[lane];
                    Scalar norm_val = norm[b];

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

    if (singular_count == Nbatch * Nf) {
        throw nb::value_error(
           "[nifty-ls finufft] Error: All systems were singular during chi2 power computation."
        );
    }
}
