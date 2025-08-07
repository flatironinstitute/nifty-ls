/* Chi2 Heterobatch implementation for multi-series processing
 * with different lengths and multiple harmonic terms
 */

#include <algorithm>
#include <complex>
#include <finufft.h>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "chi2_helpers.hpp"
#include "finufft_wrapper.hpp"
#include "utils_helpers.hpp"

namespace nb = nanobind;
using namespace nb::literals;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

template <typename Scalar>
using Complex = std::complex<Scalar>;

using utils_helpers::NormKind;
using utils_helpers::TermType;

// Function to perform Chi2 processing for a single time series
template <typename Scalar>
void process_chi2_single_series(
   const Scalar *t,   // (N_d)
   const Scalar *y,   // (N_batch, N_d)
   const Scalar *dy,  // (N_batch, N_d) / (N_batch) / nullptr
   const bool center_data,
   const bool fit_mean,
   const Scalar fmin,
   const Scalar df,
   const size_t Nf,
   const NormKind norm_kind,
   Scalar *power,  // (N_batch, Nf)
   const size_t N_batch,
   const size_t N_d,
   const size_t nterms,
   const double eps,
   finufft_opts *opts
) {
    // Set threads to 1 for single series processing
    int nthreads = 1;

    // Calculate size of trig matrices based on number of terms
    const size_t nSW = 2 * nterms + 1;
    const size_t nSY = nterms + 1;

    // Pre-allocate arrays for processing
    std::vector<Scalar> t1_(N_d), norm_(N_batch, 0), w2s_(N_batch, 0), yws_(N_batch, 0);
    std::vector<Complex<Scalar>> yw_(N_batch * N_d), w_(N_batch * N_d);
    std::vector<Scalar> Sw_(N_batch * nSW * Nf, 0), Cw_(N_batch * nSW * Nf, 0);
    std::vector<Scalar> Syw_(N_batch * nSY * Nf, 0), Cyw_(N_batch * nSY * Nf, 0);

    // Use process_chi2_inputs_raw from chi2_helpers.hpp for preprocessing
    process_chi2_inputs_raw<Scalar>(
       t1_.data(),
       yw_.data(),
       w_.data(),
       w2s_.data(),
       norm_.data(),
       yws_.data(),
       Sw_.data(),   // (Nbatch, nSW, Nf)
       Cw_.data(),   // (Nbatch, nSW, Nf)
       Syw_.data(),  // (Nbatch, nSY, Nf)
       Cyw_.data(),  // (Nbatch, nSY, Nf)
       t,            // input
       y,            // input
       dy,           // input
       N_batch,
       N_d,
       Nf,
       nSW,
       nSY,
       df,
       center_data,
       fit_mean,
       nthreads
    );

    // FINUFFT plans for different harmonics
    typename finufft_plan_type<Scalar>::type plan_yw;
    int64_t nmodes[] = {static_cast<int64_t>(Nf)};
    int plan_yw_ier =
       _finufft_makeplan<Scalar>(1, 1, nmodes, +1, 2 * N_batch, eps, &plan_yw, opts);

    if (plan_yw_ier != 0) {
        throw std::runtime_error(
           "finufft_makeplan(plan_yw) failed with error code "
           + std::to_string(plan_yw_ier)
        );
    }

    typename finufft_plan_type<Scalar>::type plan_w;
    int plan_w_ier =
       _finufft_makeplan<Scalar>(1, 1, nmodes, +1, N_batch, eps, &plan_w, opts);

    if (plan_w_ier != 0) {
        throw std::runtime_error(
           "finufft_makeplan(plan_w) failed with error code "
           + std::to_string(plan_w_ier)
        );
    }

    // Combined input array for yw and w
    std::vector<Complex<Scalar>> yw_w_(2 * N_batch * N_d);
    std::vector<Scalar> tj(N_d);
    std::vector<Complex<Scalar>> yw_w_j(2 * N_batch * N_d);
    std::vector<Complex<Scalar>> w_j(N_batch * N_d);

    // Phase shift factor
    const Scalar factor = Scalar(Nf / 2) + fmin / df;

    // Copy yw and w into combined array for combined processing
    for (size_t i = 0; i < N_batch; ++i) {
        for (size_t n = 0; n < N_d; ++n) {
            yw_w_[i * N_d + n]                 = yw_[i * N_d + n];
            yw_w_[N_batch * N_d + i * N_d + n] = w_[i * N_d + n];
        }
    }

    // Loop over harmonics from 1 to nterms (inclusive)
    for (size_t j = 1; j < nterms + 1; ++j) {
        compute_t_raw<Scalar>(
           t1_.data(),
           yw_w_.data(),
           j,
           N_d,
           2 * N_batch,
           factor,
           tj.data(),
           yw_w_j.data(),
           nthreads
        );

        // Set points and execute plan for combined transform
        int setpts_ier = _finufft_setpts<Scalar>(
           plan_yw, N_d, tj.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
        );

        if (setpts_ier != 0) {
            throw std::runtime_error(
               "finufft_setpts failed with error code " + std::to_string(setpts_ier)
            );
        }

        // Execute transform for yw and w combined
        std::vector<Complex<Scalar>> f1_fw(2 * N_batch * Nf);
        int exec_ier = _finufft_execute<Scalar>(plan_yw, yw_w_j.data(), f1_fw.data());

        if (exec_ier != 0) {
            throw std::runtime_error(
               "finufft_execute failed with error code " + std::to_string(exec_ier)
            );
        }

        // Store results in trig matrices
        for (size_t i = 0; i < N_batch; ++i) {
            for (size_t f = 0; f < Nf; ++f) {
                // Syw_[i, j, f] = f1_fw[i,f]
                Syw_[i * nSY * Nf + j * Nf + f] = f1_fw[i * Nf + f].imag();
                Cyw_[i * nSY * Nf + j * Nf + f] = f1_fw[i * Nf + f].real();
                Sw_[i * nSW * Nf + j * Nf + f]  = f1_fw[(N_batch + i) * Nf + f].imag();
                Cw_[i * nSW * Nf + j * Nf + f]  = f1_fw[(N_batch + i) * Nf + f].real();
            }
        }
    }

    // Loop for additional terms needed for w only (j = nterms+1 to 2*nterms)
    for (size_t j = nterms + 1; j < 2 * nterms + 1; ++j) {
        compute_t_raw<Scalar>(
           t1_.data(),
           w_.data(),
           j,
           N_d,
           N_batch,
           factor,
           tj.data(),   // ti in py
           w_j.data(),  // yw_w_i in py
           nthreads
        );

        // Set points and execute plan for w only
        int setpts_ier = _finufft_setpts<Scalar>(
           plan_w, N_d, tj.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
        );

        if (setpts_ier != 0) {
            throw std::runtime_error(
               "finufft_setpts failed with error code " + std::to_string(setpts_ier)
            );
        }

        // Execute transform for w only
        std::vector<Complex<Scalar>> f2_all(N_batch * Nf);
        int exec_ier = _finufft_execute<Scalar>(plan_w, w_j.data(), f2_all.data());

        if (exec_ier != 0) {
            throw std::runtime_error(
               "finufft_execute failed with error code " + std::to_string(exec_ier)
            );
        }

        // Store results in trig matrices
        for (size_t i = 0; i < N_batch; ++i) {
            for (size_t f = 0; f < Nf; ++f) {
                // Sw_[i, j, f] = f2_all[i,f]
                Sw_[i * nSW * Nf + j * Nf + f] = f2_all[i * Nf + f].imag();
                Cw_[i * nSW * Nf + j * Nf + f] = f2_all[i * Nf + f].real();
            }
        }
    }

    // Clean up finufft plans
    _finufft_destroy<Scalar>(plan_yw);
    _finufft_destroy<Scalar>(plan_w);

    // Build the order list for chi2 fitting
    std::vector<TermType> order_types;
    std::vector<size_t> order_indices;

    if (fit_mean) {
        order_types.push_back(TermType::Cosine);
        order_indices.push_back(0);
    }

    for (size_t i = 1; i <= nterms; ++i) {
        order_types.push_back(TermType::Sine);
        order_indices.push_back(i);
        order_types.push_back(TermType::Cosine);
        order_indices.push_back(i);
    }

    // Use process_chi2_outputs_raw from chi2_helpers.hpp for postprocessing
    process_chi2_outputs_raw<Scalar>(
       power,  // (Nbatch, Nf)
       Sw_.data(),
       Cw_.data(),
       Syw_.data(),
       Cyw_.data(),
       norm_.data(),
       order_types,
       order_indices,
       N_batch,
       nSW,
       nSY,
       Nf,
       norm_kind,
       nthreads
    );
}

template <typename Scalar>
void process_chi2_hetero_batch(
   const std::vector<nifty_arr_1d<const Scalar>> &t_list,
   const std::vector<nifty_arr_2d<const Scalar>> &y_list,
   const std::optional<std::vector<nifty_arr_2d<const Scalar>>> &dy_list,
   const std::vector<Scalar> &fmin_list,
   const std::vector<Scalar> &df_list,
   const std::vector<size_t> &Nf_list,
   std::vector<nifty_arr_2d<Scalar>> &powers,  // output
   const std::string &normalization,
   int nthreads,
   const bool center_data,
   const bool fit_mean,
   const double eps,
   const double upsampfac,
   const int fftw,
   const size_t nterms,
   const bool verbose
) {
#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
    if (nthreads > omp_get_max_threads()) {
        fprintf(
           stderr,
           "[nifty-ls chi2_heterobatch] Warning: nthreads (%d) > omp_get_max_threads() (%d). Performance may be suboptimal.\n",
           nthreads,
           omp_get_max_threads()
        );
    }
#else
    (void) nthreads;  // suppress unused variable warning
#endif

    const size_t N_series = t_list.size();

    // Check input sizes
    if (y_list.size() != N_series || fmin_list.size() != N_series
        || df_list.size() != N_series || Nf_list.size() != N_series) {
        throw std::runtime_error("All input lists must have same length");
    }
    if (dy_list.has_value() && dy_list->size() != N_series) {
        throw std::runtime_error(
           "dy_list must have same length as t_list when provided"
        );
    }

    // Create dy_filled if dy_list is not provided
    std::vector<nifty_arr_2d<const Scalar>> dy_filled;
    if (!dy_list.has_value()) {
        dy_filled.reserve(N_series);
        for (size_t i = 0; i < N_series; ++i) {
            const auto &y_i = y_list[i];
            const auto &t_i = t_list[i];
            size_t N_batch  = y_i.shape(0);
            size_t N_d      = t_i.shape(0);

            auto *data_ptr = new Scalar[N_batch * N_d];  // (N_batch, N_d)
            std::fill(data_ptr, data_ptr + N_batch * N_d, Scalar(1));

            size_t shape[2] = {N_batch, N_d};
            nifty_arr_2d<const Scalar> dy_ones(
               data_ptr,
               2,  //(ndim)
               shape,
               nb::capsule(data_ptr, [](void *p) noexcept {
                   delete[] static_cast<Scalar *>(p);
               })
            );
            dy_filled.push_back(dy_ones);
        }
    }
    const auto &dy_values = dy_list.has_value() ? *dy_list : dy_filled;

    // Set up finufft options
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads  = 1;  // Single thread per series
    opts.debug     = verbose ? 1 : 0;
    opts.modeord   = 0;
    opts.upsampfac = upsampfac;
    opts.fftw      = fftw;

    // Normalization kind
    static const std::unordered_map<std::string, NormKind> norm_map = {
       {"standard", NormKind::Standard},
       {"model", NormKind::Model},
       {"log", NormKind::Log},
       {"psd", NormKind::PSD}
    };

    std::string norm_lower = normalization;
    std::transform(
       norm_lower.begin(), norm_lower.end(), norm_lower.begin(), [](unsigned char c) {
           return std::tolower(c);
       }
    );

    NormKind norm_kind;
    try {
        norm_kind = norm_map.at(norm_lower);
    } catch (const std::out_of_range &e) {
        throw std::invalid_argument("Unknown normalization type: " + norm_lower);
    }

    // Start OMP
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
#endif
    for (size_t i = 0; i < N_series; ++i) {
        // Data for single series
        const auto &t_i  = t_list[i];
        const auto &y_i  = y_list[i];
        const auto &dy_i = dy_values[i];
        size_t N_d       = t_i.shape(0);
        size_t N_batch   = y_i.shape(0);

        auto &power = powers[i];

        process_chi2_single_series(
           t_i.data(),
           y_i.data(),
           dy_i.data(),
           center_data,
           fit_mean,
           fmin_list[i],
           df_list[i],
           Nf_list[i],
           norm_kind,
           power.data(),
           N_batch,
           N_d,
           nterms,
           eps,
           &opts
        );

        if (verbose) {
            std::cout << "Processed chi2 series " << i << " (" << N_batch << "," << N_d
                      << ") with " << nterms << " terms\n";
        }
    }
}

NB_MODULE(finufft_chi2_heterobatch_helpers, m) {
    m.def(
       "process_chi2_hetero_batch",
       &process_chi2_hetero_batch<float>,
       "t_list"_a,
       "y_list"_a,
       "dy_list"_a.none(),
       "fmin_list"_a,
       "df_list"_a,
       "Nf_list"_a,
       "powers"_a,
       "normalization"_a,
       "nthreads"_a,
       "center_data"_a,
       "fit_mean"_a,
       "eps"_a,
       "upsampfac"_a,
       "fftw"_a,
       "nterms"_a,
       "verbose"_a
    );
    m.def(
       "process_chi2_hetero_batch",
       &process_chi2_hetero_batch<double>,
       "t_list"_a,
       "y_list"_a,
       "dy_list"_a.none(),
       "fmin_list"_a,
       "df_list"_a,
       "Nf_list"_a,
       "powers"_a,
       "normalization"_a,
       "nthreads"_a,
       "center_data"_a,
       "fit_mean"_a,
       "eps"_a,
       "upsampfac"_a,
       "fftw"_a,
       "nterms"_a,
       "verbose"_a
    );
}