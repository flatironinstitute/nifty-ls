/* Heterobatch implementation for multi-series processing
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

#include "cpu_helpers.hpp"
#include "finufft_wrapper.hpp"
#include "utils_helpers.hpp"

namespace nb = nanobind;
using namespace nb::literals;

template <typename Scalar>
using nifty_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cpu>;

template <typename Scalar>
using nifty_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cpu>;

using utils_helpers::NormKind;

template <typename Scalar>
using Complex = std::complex<Scalar>;

template <typename Scalar>
void process_single_series(
   const Scalar *t,   // (N_d)
   const Scalar *y,   // (N_batch, N_d)
   const Scalar *dy,  // (N_batch, N_d) / (N_batch) / nullptr
   const bool broadcast_dy,
   const bool center_data,
   const bool fit_mean,
   const Scalar fmin,
   const Scalar df,
   const size_t Nf,
   const NormKind norm_kind,
   Scalar *power,  // (N_batch, Nf)
   const size_t N_batch,
   const size_t N_d,
   const double eps,
   finufft_opts *opts
) {
    std::vector<Scalar> t1_(N_d), t2_(N_d), norm_(N_batch);
    std::vector<Complex<Scalar>> yw_(N_batch * N_d), w_(N_batch * N_d),
       w2_(N_batch * N_d);
    const bool psd_norm = (norm_kind == NormKind::PSD);

    // Set threads to 1 for small problem size
    int nthreads = 1;

    process_finufft_inputs_raw(
       t1_.data(),
       t2_.data(),
       yw_.data(),
       w_.data(),
       w2_.data(),
       norm_.data(),
       t,   // input
       y,   // input
       dy,  // input
       broadcast_dy,
       fmin,
       df,
       Nf,
       center_data,
       fit_mean,
       psd_norm,
       nthreads,
       N_batch,
       N_d
    );

    // Finufft
    // Create yw_w
    std::vector<Complex<Scalar>> yw_w_(2 * N_batch * N_d);
    std::copy(yw_.begin(), yw_.end(), yw_w_.begin());
    std::copy(w_.begin(), w_.end(), yw_w_.begin() + yw_.size());

    // Set up Transform type and dim
    int type = 1;  // Type-1 NUFFT
    int dim  = 1;  // 1D

    std::vector<Complex<Scalar>> f1_(N_batch * Nf), fw_(N_batch * Nf),
       f2_(N_batch * Nf);

    // Plan solo
    finufft_plan solo_plan;
    int64_t nmodes[] = {static_cast<int64_t>(Nf)};
    int ntrans       = N_batch;
    int solo_ier =
       _finufft_makeplan<Scalar>(type, dim, nmodes, +1, ntrans, eps, &solo_plan, opts);
    if (solo_ier != 0) {
        throw std::runtime_error(
           "finufft_makeplan(solo) failed with error code " + std::to_string(solo_ier)
        );
    }

    if (fit_mean) {
        // Plan pair
        finufft_plan pair_plan;
        int ntrans   = 2 * N_batch;
        int pair_ier = _finufft_makeplan<Scalar>(
           type, dim, nmodes, +1, ntrans, eps, &pair_plan, opts
        );
        if (pair_ier != 0) {
            throw std::runtime_error(
               "finufft_makeplan(pair) failed with error code "
               + std::to_string(pair_ier)
            );
        }

        // setpts (pair)
        int setpts_ier = _finufft_setpts<Scalar>(
           pair_plan, N_d, t1_.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
        );
        if (setpts_ier != 0) {
            throw std::runtime_error(
               "finufft_setpts(pair) failed with error code "
               + std::to_string(setpts_ier)
            );
        }

        // execute (pair)
        std::vector<Complex<Scalar>> f1_fw(2 * N_batch * Nf);
        int exec_ier = _finufft_execute<Scalar>(pair_plan, yw_w_.data(), f1_fw.data());
        if (exec_ier != 0) {
            throw std::runtime_error(
               "finufft_execute(pair) failed with error code "
               + std::to_string(exec_ier)
            );
        }

        // Save and clean
        std::copy(f1_fw.begin(), f1_fw.begin() + N_batch * Nf, f1_.begin());
        std::copy(f1_fw.begin() + N_batch * Nf, f1_fw.end(), fw_.begin());
        _finufft_destroy<Scalar>(pair_plan);

    } else {
        // setpts (solo)
        int setpts_ier = _finufft_setpts<Scalar>(
           solo_plan, N_d, t1_.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
        );
        if (setpts_ier != 0) {
            throw std::runtime_error(
               "finufft_setpts(solo) failed with error code "
               + std::to_string(setpts_ier)
            );
        }

        // execute (solo)
        std::vector<Complex<Scalar>> f1_fw(N_batch * Nf);
        int exec_ier = _finufft_execute<Scalar>(solo_plan, yw_w_.data(), f1_fw.data());
        if (exec_ier != 0) {
            throw std::runtime_error(
               "finufft_execute(solo) failed with error code "
               + std::to_string(exec_ier)
            );
        }

        // Save
        std::copy(f1_fw.begin(), f1_fw.end(), f1_.begin());
        std::fill(fw_.begin(), fw_.end(), Complex<Scalar>(0.0, 0.0));
    }

    // second transform
    int setpts_ier = _finufft_setpts<Scalar>(
       solo_plan, N_d, t2_.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr
    );
    if (setpts_ier != 0) {
        throw std::runtime_error(
           "finufft_setpts(solo) failed with error code " + std::to_string(setpts_ier)
        );
    }
    int exec_ier = _finufft_execute<Scalar>(solo_plan, w2_.data(), f2_.data());
    if (exec_ier != 0) {
        throw std::runtime_error(
           "finufft_execute(second solo) failed with error code "
           + std::to_string(exec_ier)
        );
    }

    // Clean
    _finufft_destroy<Scalar>(solo_plan);

    process_finufft_outputs_raw(
       power,
       f1_.data(),
       fw_.data(),
       f2_.data(),
       norm_.data(),
       norm_kind,
       fit_mean,
       nthreads,
       N_batch,
       Nf
    );
}

template <typename Scalar>
void process_hetero_batch(
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
   const bool verbose
) {
#ifdef _OPENMP
    if (nthreads < 1) { nthreads = omp_get_max_threads(); }
    if (nthreads > omp_get_max_threads()) {
        fprintf(
           stderr,
           "[nifty-ls finufft_heterobatch] Warning: nthreads (%d) > omp_get_max_threads() (%d). Performance may be suboptimal.\n",
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
        const auto &t_i = t_list[i];
        const auto &y_i = y_list[i];
        size_t N_d      = t_i.shape(0);
        size_t N_batch  = y_i.shape(0);
        std::vector<Scalar> default_dy(N_batch, 1.0);  // (N_batch, 1)
        const Scalar *dy_ptr = default_dy.data();
        bool broadcast_dy    = true;
        if (dy_list.has_value()) {
            const auto &dy_i = dy_list.value()[i];
            dy_ptr           = dy_i.data();
            broadcast_dy     = (dy_i.shape(1) == 1);
        }
        auto &power = powers[i];

        process_single_series(
           t_i.data(),
           y_i.data(),
           dy_ptr,
           broadcast_dy,
           center_data,
           fit_mean,
           fmin_list[i],
           df_list[i],
           Nf_list[i],
           norm_kind,
           power.data(),
           N_batch,
           N_d,
           eps,
           &opts
        );

        if (verbose) {
            std::cout << "Processed series " << i << " (" << N_batch << "," << N_d
                      << ")\n";
        }
    }
}

NB_MODULE(heterobatch_helpers, m) {
    m.def(
       "process_hetero_batch",
       &process_hetero_batch<float>,
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
       "verbose"_a
    );
    m.def(
       "process_hetero_batch",
       &process_hetero_batch<double>,
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
       "verbose"_a
    );
}