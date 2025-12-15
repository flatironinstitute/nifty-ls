// CUDA implementation of chi2 preprocessing (weights, norms, base trig terms).
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <thrust/complex.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "utils_helpers.hpp"
#include <cufinufft.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

    constexpr int TPB = 256;
#define CUFINUFFT_CHECK(expr, msg)                          \
    do {                                                    \
        int _err = (expr);                                  \
        if (_err != 0) { throw std::runtime_error((msg)); } \
    } while (0)
#define SHARED_REDUCTION(type, var)                               \
    sdata[threadIdx.x] = var;                                     \
    __syncthreads();                                              \
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) { \
        if (threadIdx.x < offset) {                               \
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];    \
        }                                                         \
        __syncthreads();                                          \
    }                                                             \
    var = sdata[0];
#ifndef CUFINUFFT_TYPE1
#define CUFINUFFT_TYPE1 1
#endif

#define CUDA_CHECK(expr)                                        \
    do {                                                        \
        cudaError_t _err = (expr);                              \
        if (_err != cudaSuccess) {                              \
            throw std::runtime_error(cudaGetErrorString(_err)); \
        }                                                       \
    } while (0)

#define CUBLAS_CHECK(expr)                                      \
    do {                                                        \
        cublasStatus_t _err = (expr);                           \
        if (_err != CUBLAS_STATUS_SUCCESS) {                    \
            throw std::runtime_error("cublas error");           \
        }                                                       \
    } while (0)

    // Mirror of TermType from utils_helpers (Sine/Cosine)
    enum class TermTypeCUDA : int {
        Sine   = 0,
        Cosine = 1
    };

    inline utils_helpers::NormKind to_norm_kind(int norm_kind) {
        switch (norm_kind) {
            case 0:
                return utils_helpers::NormKind::Standard;
            case 1:
                return utils_helpers::NormKind::Model;
            case 2:
                return utils_helpers::NormKind::Log;
            case 3:
                return utils_helpers::NormKind::PSD;
            default:
                return utils_helpers::NormKind::Standard;
        }
    }

    template <typename Scalar>
    using Complex = thrust::complex<Scalar>;

    inline int clamp_block_dim_host(int block_dim) {
        if (block_dim <= 0) { return TPB; }
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        return std::min(block_dim, prop.maxThreadsPerBlock);
    }

    template <typename Scalar>
    __global__ void phase_shift_yw_w_inputs(
       const Scalar *t1,
       Scalar time_shift,
       Scalar factor,
       const Scalar *yw,
       const Scalar *w,
       Complex<Scalar> *output,
       Scalar *tn_out,
       size_t N,
       size_t Nbatch
    ) {
        const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= N) { return; }
        const Scalar tn             = time_shift * t1[j];
        tn_out[j]                   = tn;
        const Complex<Scalar> phase = exp(Complex<Scalar>(Scalar(0), factor * tn));
        for (size_t b = 0; b < Nbatch; ++b) {
            output[b * N + j] =
               Complex<Scalar>(yw[b * N + j], Scalar(0)) * phase;
            output[(b + Nbatch) * N + j] =
               Complex<Scalar>(w[b * N + j], Scalar(0)) * phase;
        }
    }

    template <typename Scalar>
    __global__ void phase_shift_w_inputs(
       const Scalar *t1,
       Scalar time_shift,
       Scalar factor,
       const Scalar *w,
       Complex<Scalar> *output,
       Scalar *tn_out,
       size_t N,
       size_t Nbatch
    ) {
        const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= N) { return; }
        const Scalar tn             = time_shift * t1[j];
        tn_out[j]                   = tn;
        const Complex<Scalar> phase = exp(Complex<Scalar>(Scalar(0), factor * tn));
        for (size_t b = 0; b < Nbatch; ++b) {
            output[b * N + j] = Complex<Scalar>(w[b * N + j], Scalar(0)) * phase;
        }
    }

    template <typename Scalar>
    __global__ void scatter_first_pass(
       const Complex<Scalar> *f1fw,
       Scalar *Sw,
       Scalar *Cw,
       Scalar *Syw,
       Scalar *Cyw,
       int term_idx,
       size_t Nf,
       size_t Nbatch,
       int nSW,
       int nSY
    ) {
        const size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t total = Nbatch * Nf;
        if (idx >= total) { return; }
        const size_t batch = idx / Nf;
        const size_t f     = idx - batch * Nf;

        const Complex<Scalar> ywv = f1fw[(batch + static_cast<size_t>(0)) * Nf + f];
        const Complex<Scalar> wv  = f1fw[(batch + Nbatch) * Nf + f];

        const size_t base_sw  = batch * static_cast<size_t>(nSW) * Nf;
        const size_t base_sy  = batch * static_cast<size_t>(nSY) * Nf;
        const size_t offset_w = base_sw + static_cast<size_t>(term_idx) * Nf + f;
        const size_t offset_y = base_sy + static_cast<size_t>(term_idx) * Nf + f;

        Sw[offset_w]  = wv.imag();
        Cw[offset_w]  = wv.real();
        Syw[offset_y] = ywv.imag();
        Cyw[offset_y] = ywv.real();
    }

    template <typename Scalar>
    __global__ void scatter_second_pass(
       const Complex<Scalar> *f2,
       Scalar *Sw,
       Scalar *Cw,
       int term_idx,
       size_t Nf,
       size_t Nbatch,
       int nSW
    ) {
        const size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t total = Nbatch * Nf;
        if (idx >= total) { return; }
        const size_t batch = idx / Nf;
        const size_t f     = idx - batch * Nf;

        const size_t base_sw     = batch * static_cast<size_t>(nSW) * Nf;
        const size_t offset_w    = base_sw + static_cast<size_t>(term_idx) * Nf + f;
        const Complex<Scalar> wv = f2[batch * Nf + f];
        Sw[offset_w]             = wv.imag();
        Cw[offset_w]             = wv.real();
    }

    template <typename Scalar>
    __global__ void
    set_t1_kernel(Scalar *t1, const Scalar *t, const Scalar df, const size_t N) {
        const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= N) { return; }
        t1[j] = static_cast<Scalar>(2 * CUDART_PI) * df * t[j];
    }

    template <typename Scalar>
    __global__ void process_chi2_inputs_kernel(
       Scalar *w2s,
       Scalar *norm,
       Scalar *yws,
       Scalar *Sw,
       Scalar *Cw,
       Scalar *Syw,
       Scalar *Cyw,
       Scalar *yw,
       Scalar *w,
       const Scalar *y,
       const Scalar *dy,
       const bool center_data,
       const bool fit_mean,
       const size_t N,
       const size_t Nbatch,
       const size_t Nf,
       const size_t nSW,
       const size_t nSY
    ) {
        const size_t batch = blockIdx.x;
        if (batch >= Nbatch) { return; }
        extern __shared__ double sdata[];

        const size_t base = batch * N;

        // Reduction for sum_w and yoff
        double local_wsum = 0.0;
        double local_yoff = 0.0;
        for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
            const Scalar d  = dy[base + j];
            const Scalar wt = Scalar(1) / (d * d);
            local_wsum += static_cast<double>(wt);
            local_yoff += static_cast<double>(wt * y[base + j]);
        }
        SHARED_REDUCTION(double, local_wsum);
        double wsum = local_wsum;
        if (center_data || fit_mean) {
            SHARED_REDUCTION(double, local_yoff);
            local_yoff /= wsum;
        } else {
            local_yoff = 0.0;
        }
        Scalar yoff = static_cast<Scalar>(local_yoff);

        // Second reduction for norm and yws; also fill yw/w
        double local_norm = 0.0;
        double local_yws  = 0.0;
        for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
            const Scalar d  = dy[base + j];
            const Scalar wt = Scalar(1) / (d * d);
            const Scalar ym = y[base + j] - yoff;
            local_norm += static_cast<double>(wt * ym * ym);
            local_yws += static_cast<double>(ym * wt);
            yw[base + j] = ym * wt;
            w[base + j]  = wt;
        }
        SHARED_REDUCTION(double, local_norm);
        SHARED_REDUCTION(double, local_yws);

        if (threadIdx.x == 0) {
            w2s[batch]  = static_cast<Scalar>(wsum);
            norm[batch] = static_cast<Scalar>(local_norm);
            yws[batch] =
               (center_data || fit_mean) ? Scalar(0) : static_cast<Scalar>(local_yws);
        }
        __syncthreads();

        // Parallel init for trig matrices base terms
        for (size_t f = threadIdx.x; f < Nf; f += blockDim.x) {
            const size_t idx_sw = batch * nSW * Nf + f;
            const size_t idx_sy = batch * nSY * Nf + f;
            Sw[idx_sw]          = Scalar(0);
            Syw[idx_sy]         = Scalar(0);
            Cw[idx_sw]          = static_cast<Scalar>(wsum);
            Cyw[idx_sy] =
               (center_data || fit_mean) ? Scalar(0) : static_cast<Scalar>(local_yws);
        }
    }

    // cufinufft type-1 plan helpers
    template <typename Scalar>
    struct CufinufftTraits;

    template <>
    struct CufinufftTraits<double> {
        using plan_t    = cufinufft_plan;
        using complex_t = cuDoubleComplex;

        static int makeplan(
           int64_t dim,
           int64_t *nmodes,
           int iflag,
           int64_t ntrans,
           double eps,
           plan_t *plan,
           cufinufft_opts *opts
        ) {
            return cufinufft_makeplan(
               CUFINUFFT_TYPE1, dim, nmodes, iflag, ntrans, eps, plan, opts
            );
        }

        static int setpts(plan_t plan, int64_t M, const double *x) {
            return cufinufft_setpts(
               plan,
               M,
               const_cast<double *>(x),
               nullptr,
               nullptr,
               0,
               nullptr,
               nullptr,
               nullptr
            );
        }

        static int execute(plan_t plan, const complex_t *c, complex_t *fk) {
            return cufinufft_execute(plan, const_cast<complex_t *>(c), fk);
        }

        static int destroy(plan_t plan) { return cufinufft_destroy(plan); }
    };

    template <>
    struct CufinufftTraits<float> {
        using plan_t    = cufinufftf_plan;
        using complex_t = cuFloatComplex;

        static int makeplan(
           int64_t dim,
           int64_t *nmodes,
           int iflag,
           int64_t ntrans,
           double eps,
           plan_t *plan,
           cufinufft_opts *opts
        ) {
            return cufinufftf_makeplan(
               CUFINUFFT_TYPE1, dim, nmodes, iflag, ntrans, eps, plan, opts
            );
        }

        static int setpts(plan_t plan, int64_t M, const float *x) {
            return cufinufftf_setpts(
               plan,
               M,
               const_cast<float *>(x),
               nullptr,
               nullptr,
               0,
               nullptr,
               nullptr,
               nullptr
            );
        }

        static int execute(plan_t plan, const complex_t *c, complex_t *fk) {
            return cufinufftf_execute(plan, const_cast<complex_t *>(c), fk);
        }

        static int destroy(plan_t plan) { return cufinufftf_destroy(plan); }
    };

    // Kernel to assemble XTX and XTy for each (batch, freq)
    template <typename Scalar>
    __global__ void assemble_xtx_xty(
       Scalar *A,         // [batch*chunk, k, k], column-major
       Scalar *B,         // [batch*chunk, k]
       const Scalar *Sw,  // [Nbatch, nSW, Nf_total]
       const Scalar *Cw,
       const Scalar *Syw,  // [Nbatch, nSY, Nf_total]
       const Scalar *Cyw,
       const Scalar *norm,  // [Nbatch]
       const int *order_types,
       const int *order_indices,
       int k,
       int nSW,
       int nSY,
       size_t Nf_total,
       size_t Nbatch,
       int start_f,
       int chunk_f
    ) {
        const size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t total = static_cast<size_t>(Nbatch) * static_cast<size_t>(chunk_f);
        if (idx >= total) return;
        const size_t batch    = idx / static_cast<size_t>(chunk_f);
        const size_t f_local  = idx - batch * static_cast<size_t>(chunk_f);
        const size_t f        = static_cast<size_t>(start_f) + f_local;
        const size_t Nf_total_sz = Nf_total;

        const auto Sw_b  = Sw + batch * nSW * Nf_total_sz;
        const auto Cw_b  = Cw + batch * nSW * Nf_total_sz;
        const auto Syw_b = Syw + batch * nSY * Nf_total_sz;
        const auto Cyw_b = Cyw + batch * nSY * Nf_total_sz;

        auto A_mat = A + idx * k * k;  // column-major
        auto B_vec = B + idx * k;

        auto get_S  = [&](int m) -> Scalar { return Syw_b[m * Nf_total_sz + f]; };
        auto get_C  = [&](int m) -> Scalar { return Cyw_b[m * Nf_total_sz + f]; };
        auto get_SS = [&](int m, int n) -> Scalar {
            int diff = abs(m - n);
            return Scalar(0.5)
                   * (Cw_b[diff * Nf_total_sz + f]
                      - Cw_b[(m + n) * Nf_total_sz + f]);
        };
        auto get_CC = [&](int m, int n) -> Scalar {
            int diff = abs(m - n);
            return Scalar(0.5)
                   * (Cw_b[diff * Nf_total_sz + f]
                      + Cw_b[(m + n) * Nf_total_sz + f]);
        };
        auto get_SC = [&](int m, int n) -> Scalar {
            int diff    = abs(m - n);
            Scalar term =
               (m >= n ? Scalar(1) : Scalar(-1)) * Sw_b[diff * Nf_total_sz + f];
            return Scalar(0.5) * (term + Sw_b[(m + n) * Nf_total_sz + f]);
        };
        auto get_CS = [&](int m, int n) -> Scalar {
            int diff    = abs(n - m);
            Scalar term =
               (n >= m ? Scalar(1) : Scalar(-1)) * Sw_b[diff * Nf_total_sz + f];
            return Scalar(0.5) * (term + Sw_b[(n + m) * Nf_total_sz + f]);
        };

        // Fill XTy
        for (int i = 0; i < k; ++i) {
            const int t = order_types[i];
            const int m = order_indices[i];
            B_vec[i] =
               (t == static_cast<int>(TermTypeCUDA::Cosine)) ? get_C(m) : get_S(m);
        }

        // Fill XTX (column-major)
        for (int col = 0; col < k; ++col) {
            const int tcol = order_types[col];
            const int mcol = order_indices[col];
            for (int row = 0; row < k; ++row) {
                const int trow = order_types[row];
                const int mrow = order_indices[row];
                Scalar val     = Scalar(0);
                if (trow == static_cast<int>(TermTypeCUDA::Cosine)
                    && tcol == static_cast<int>(TermTypeCUDA::Cosine)) {
                    val = get_CC(mrow, mcol);
                } else if (trow == static_cast<int>(TermTypeCUDA::Cosine)
                           && tcol == static_cast<int>(TermTypeCUDA::Sine)) {
                    val = get_CS(mrow, mcol);
                } else if (trow == static_cast<int>(TermTypeCUDA::Sine)
                           && tcol == static_cast<int>(TermTypeCUDA::Cosine)) {
                    val = get_SC(mrow, mcol);
                } else {  // Sine, Sine
                    val = get_SS(mrow, mcol);
                }
                A_mat[col * k + row] = val;  // column-major
            }
        }
    }

    // Kernel to compute power from solution and XTy
    template <typename Scalar>
    __global__ void compute_power_kernel(
       Scalar *power,               // full output [Nbatch, Nf_total]
       const Scalar *solution,      // B after solve, size chunk_total*k
       const Scalar *XTy,           // original XTy, size chunk_total*k
       const Scalar *norm,          // [Nbatch]
       int k,
       size_t Nf_total,
       size_t Nbatch,
       int start_f,
       int chunk_f,
       utils_helpers::NormKind norm_kind
    ) {
        const size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t total = static_cast<size_t>(Nbatch) * static_cast<size_t>(chunk_f);
        if (idx >= total) return;
        const size_t batch = idx / static_cast<size_t>(chunk_f);
        const size_t f_local = idx - batch * static_cast<size_t>(chunk_f);
        const size_t f       = static_cast<size_t>(start_f) + f_local;

        const Scalar *beta = solution + idx * k;
        const Scalar *xty  = XTy + idx * k;

        Scalar raw_power = Scalar(0);
        for (int i = 0; i < k; ++i) { raw_power += xty[i] * beta[i]; }

        Scalar norm_val = norm[batch];
        Scalar out      = raw_power;
        switch (norm_kind) {
            case utils_helpers::NormKind::Standard:
                out = raw_power / norm_val;
                break;
            case utils_helpers::NormKind::Model:
                out = raw_power / (norm_val - raw_power);
                break;
            case utils_helpers::NormKind::Log:
                out = -log(Scalar(1) - raw_power / norm_val);
                break;
            case utils_helpers::NormKind::PSD:
                out = raw_power * Scalar(0.5);
                break;
        }
        power[batch * Nf_total + f] = out;
    }

    template <typename Scalar>
    __global__ void set_batched_ptrs(
       Scalar **A_array,
       Scalar **B_array,
       Scalar *A,
       Scalar *B,
       int stride_A,
       int stride_B,
       size_t n
    ) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) { return; }
        A_array[idx] = A + idx * static_cast<size_t>(stride_A);
        B_array[idx] = B + idx * static_cast<size_t>(stride_B);
    }

    template <typename Scalar>
    void process_chi2_outputs_device_impl(
       Scalar *d_power,
       const Scalar *d_Sw,
       const Scalar *d_Cw,
       const Scalar *d_Syw,
       const Scalar *d_Cyw,
       const Scalar *d_norm,
       const int *d_order_types,
       const int *d_order_indices,
       int k,
       int nSW,
       int nSY,
       size_t Nf_total,
       size_t Nbatch,
       int norm_kind_int
    ) {
        // Process frequencies in chunks to limit memory and cublas batch size.
        constexpr int CHUNK_F = 8192;
        const size_t max_chunk_f =
           static_cast<size_t>(CHUNK_F) < Nf_total ? static_cast<size_t>(CHUNK_F)
                                                   : Nf_total;
        const size_t max_chunk_total = static_cast<size_t>(Nbatch) * max_chunk_f;

        Scalar *d_A   = nullptr;
        Scalar *d_B   = nullptr;
        Scalar *d_XTy = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, max_chunk_total * k * k * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_B, max_chunk_total * k * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_XTy, max_chunk_total * k * sizeof(Scalar)));

        int *d_pivots = nullptr;
        int *d_infoArray = nullptr;
        CUDA_CHECK(cudaMalloc(&d_pivots, max_chunk_total * k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_infoArray, max_chunk_total * sizeof(int)));

        Scalar **d_A_array = nullptr;
        Scalar **d_B_array = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A_array, max_chunk_total * sizeof(Scalar *)));
        CUDA_CHECK(cudaMalloc(&d_B_array, max_chunk_total * sizeof(Scalar *)));

        {
            const int threads = 256;
            const dim3 grid((max_chunk_total + threads - 1) / threads);
            set_batched_ptrs<<<grid, threads>>>(
               d_A_array, d_B_array, d_A, d_B, k * k, k, max_chunk_total
            );
            CUDA_CHECK(cudaGetLastError());
        }

        cublasHandle_t cublas_handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        const int tpb = 128;
        for (size_t start_f = 0; start_f < Nf_total; start_f += max_chunk_f) {
            const size_t chunk_f =
               std::min(max_chunk_f, static_cast<size_t>(Nf_total - start_f));
            const size_t chunk_total =
               static_cast<size_t>(Nbatch) * static_cast<size_t>(chunk_f);

            const dim3 grid((chunk_total + tpb - 1) / tpb);
            assemble_xtx_xty<<<grid, tpb>>>(
               d_A,
               d_B,
               d_Sw,
               d_Cw,
               d_Syw,
               d_Cyw,
               d_norm,
               d_order_types,
               d_order_indices,
               k,
               nSW,
               nSY,
               Nf_total,
               Nbatch,
               static_cast<int>(start_f),
               static_cast<int>(chunk_f)
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(
               d_XTy,
               d_B,
               chunk_total * k * sizeof(Scalar),
               cudaMemcpyDeviceToDevice
            ));

            const int batch_count = static_cast<int>(chunk_total);
            if constexpr (std::is_same<Scalar, float>::value) {
                CUBLAS_CHECK(cublasSgetrfBatched(
                   cublas_handle, k, d_A_array, k, d_pivots, d_infoArray, batch_count
                ));
                int info = 0;
                CUBLAS_CHECK(cublasSgetrsBatched(
                   cublas_handle,
                   CUBLAS_OP_N,
                   k,
                   1,
                   (const float **) d_A_array,
                   k,
                   d_pivots,
                   d_B_array,
                   k,
                   &info,
                   batch_count
                ));
                if (info != 0) {
                    throw std::runtime_error("cublasSgetrsBatched failed (info != 0)");
                }
            } else {
                CUBLAS_CHECK(cublasDgetrfBatched(
                   cublas_handle, k, d_A_array, k, d_pivots, d_infoArray, batch_count
                ));
                int info = 0;
                CUBLAS_CHECK(cublasDgetrsBatched(
                   cublas_handle,
                   CUBLAS_OP_N,
                   k,
                   1,
                   (const double **) d_A_array,
                   k,
                   d_pivots,
                   d_B_array,
                   k,
                   &info,
                   batch_count
                ));
                if (info != 0) {
                    throw std::runtime_error("cublasDgetrsBatched failed (info != 0)");
                }
            }
            CUDA_CHECK(cudaGetLastError());

            compute_power_kernel<<<grid, tpb>>>(
               d_power,
               d_B,
               d_XTy,
               d_norm,
               k,
               Nf_total,
               Nbatch,
               static_cast<int>(start_f),
               static_cast<int>(chunk_f),
               to_norm_kind(norm_kind_int)
            );
            CUDA_CHECK(cudaGetLastError());
        }

        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        cudaFree(d_pivots);
        cudaFree(d_infoArray);
        cudaFree(d_A_array);
        cudaFree(d_B_array);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_XTy);
    }

    // End-to-end CUDA path: preprocess, NUFFT, and postprocess fully on GPU.
    template <typename Scalar>
    nb::ndarray<Scalar, nb::ndim<2>, nb::numpy> lombscargle_chi2_cuda(
       nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cpu> t_,
       nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cpu> y_,
       nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cpu> dy_,
       const Scalar fmin,
       const Scalar df,
       const size_t Nf,
       const bool center_data,
       const bool fit_mean,
       const utils_helpers::NormKind norm_kind,
       const size_t nterms,
       const double eps,
       const int gpu_method,
       const int block_dim
    ) {
        const size_t N         = y_.shape(1);
        const size_t Nbatch    = y_.shape(0);
        const size_t NN        = Nbatch * N;
        const size_t nSW       = 2 * nterms + 1;
        const size_t nSY       = nterms + 1;
        const size_t yww_trans = 2 * Nbatch;
        // Match CPU helper: use integer division for Nf/2 to avoid 0.5 offset when Nf
        // is odd.
        const Scalar phase_factor = static_cast<Scalar>(Nf / 2) + fmin / df;
        const int threads         = clamp_block_dim_host(block_dim);

        // Device buffers
        Scalar *d_t     = nullptr;
        Scalar *d_y     = nullptr;
        Scalar *d_dy    = nullptr;
        Scalar *d_t1    = nullptr;
        Scalar *d_yw    = nullptr;
        Scalar *d_w     = nullptr;
        Scalar *d_w2s   = nullptr;
        Scalar *d_norm  = nullptr;
        Scalar *d_yws   = nullptr;
        Scalar *d_Sw    = nullptr;
        Scalar *d_Cw    = nullptr;
        Scalar *d_Syw   = nullptr;
        Scalar *d_Cyw   = nullptr;
        Scalar *d_tn    = nullptr;
        Scalar *d_power = nullptr;

        Complex<Scalar> *d_yw_w_work = nullptr;
        Complex<Scalar> *d_fk        = nullptr;

        CUDA_CHECK(cudaMalloc(&d_t, N * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_y, NN * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_dy, NN * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_t1, N * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_yw, NN * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_w, NN * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_w2s, Nbatch * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_norm, Nbatch * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_yws, Nbatch * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_Sw, Nbatch * nSW * Nf * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_Cw, Nbatch * nSW * Nf * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_Syw, Nbatch * nSY * Nf * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_Cyw, Nbatch * nSY * Nf * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_tn, N * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_power, Nbatch * Nf * sizeof(Scalar)));

        CUDA_CHECK(cudaMalloc(&d_yw_w_work, yww_trans * N * sizeof(Complex<Scalar>)));
        CUDA_CHECK(cudaMalloc(&d_fk, yww_trans * Nf * sizeof(Complex<Scalar>)));

        CUDA_CHECK(
           cudaMemcpy(d_t, t_.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice)
        );
        CUDA_CHECK(
           cudaMemcpy(d_y, y_.data(), NN * sizeof(Scalar), cudaMemcpyHostToDevice)
        );
        CUDA_CHECK(
           cudaMemcpy(d_dy, dy_.data(), NN * sizeof(Scalar), cudaMemcpyHostToDevice)
        );

        // Preprocess on GPU (weights, norms, base trig terms, t1)
        const int tpb_pre     = (block_dim > 0) ? std::min(block_dim, TPB) : TPB;
        const int tpb_clamped = (tpb_pre > TPB) ? TPB : tpb_pre;
        const dim3 grid_t1((N + threads - 1) / threads);
        set_t1_kernel<<<grid_t1, threads>>>(d_t1, d_t, df, N);
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_batch(Nbatch);
        const size_t shmem_bytes = tpb_clamped * sizeof(double);
        process_chi2_inputs_kernel<<<grid_batch, tpb_clamped, shmem_bytes>>>(
           d_w2s,
           d_norm,
           d_yws,
           d_Sw,
           d_Cw,
           d_Syw,
           d_Cyw,
           d_yw,
           d_w,
           d_y,
           d_dy,
           center_data,
           fit_mean,
           N,
           Nbatch,
           Nf,
           nSW,
           nSY
        );
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_phase((N + threads - 1) / threads);
        const dim3 grid_scatter((Nbatch * Nf + threads - 1) / threads);

        using Traits    = CufinufftTraits<Scalar>;
        using complex_t = typename Traits::complex_t;

        int iflag         = 1;
        int64_t dim       = 1;
        int64_t nmodes[1] = {static_cast<int64_t>(Nf)};
        cufinufft_opts opts;
        cufinufft_default_opts(&opts);
        opts.gpu_method = gpu_method;

        typename Traits::plan_t plan_yw = nullptr;
        typename Traits::plan_t plan_w  = nullptr;
        CUFINUFFT_CHECK(
           Traits::makeplan(
              dim,
              nmodes,
              iflag,
              static_cast<int64_t>(yww_trans),
              eps,
              &plan_yw,
              &opts
           ),
           "cufinufft_makeplan failed"
        );
        CUFINUFFT_CHECK(
           Traits::makeplan(
              dim,
              nmodes,
              iflag,
              static_cast<int64_t>(Nbatch),
              eps,
              &plan_w,
              &opts
           ),
           "cufinufft_makeplan failed"
        );

        // Harmonic loop for yw and w together
        for (size_t j = 1; j <= nterms; ++j) {
            phase_shift_yw_w_inputs<<<grid_phase, threads>>>(
               d_t1,
               static_cast<Scalar>(j),
               phase_factor,
               d_yw,
               d_w,
               d_yw_w_work,
               d_tn,
               N,
               Nbatch
            );
            CUDA_CHECK(cudaGetLastError());

            CUFINUFFT_CHECK(Traits::setpts(plan_yw, static_cast<int64_t>(N), d_tn),
                            "cufinufft_setpts failed");
            CUFINUFFT_CHECK(
               Traits::execute(
                  plan_yw,
                  reinterpret_cast<const complex_t *>(d_yw_w_work),
                  reinterpret_cast<complex_t *>(d_fk)
               ),
               "cufinufft_execute failed"
            );

            scatter_first_pass<<<grid_scatter, threads>>>(
               d_fk,
               d_Sw,
               d_Cw,
               d_Syw,
               d_Cyw,
               static_cast<int>(j),
               Nf,
               Nbatch,
               static_cast<int>(nSW),
               static_cast<int>(nSY)
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // Harmonic loop for w only (indices nterms+1 ... 2*nterms)
        for (size_t i = nterms + 1; i <= 2 * nterms; ++i) {
            phase_shift_w_inputs<<<grid_phase, threads>>>(
               d_t1,
               static_cast<Scalar>(i),
               phase_factor,
               d_w,
               d_yw_w_work,
               d_tn,
               N,
               Nbatch
            );
            CUDA_CHECK(cudaGetLastError());

            CUFINUFFT_CHECK(Traits::setpts(plan_w, static_cast<int64_t>(N), d_tn),
                            "cufinufft_setpts failed");
            CUFINUFFT_CHECK(
               Traits::execute(
                  plan_w,
                  reinterpret_cast<const complex_t *>(d_yw_w_work),
                  reinterpret_cast<complex_t *>(d_fk)
               ),
               "cufinufft_execute failed"
            );

            scatter_second_pass<<<grid_scatter, threads>>>(
               d_fk, d_Sw, d_Cw, static_cast<int>(i), Nf, Nbatch, static_cast<int>(nSW)
            );
            CUDA_CHECK(cudaGetLastError());
        }

        CUFINUFFT_CHECK(Traits::destroy(plan_yw), "cufinufft_destroy failed");
        CUFINUFFT_CHECK(Traits::destroy(plan_w), "cufinufft_destroy failed");

        // Build order arrays on host
        std::vector<int> order_types;
        std::vector<int> order_indices;
        order_types.reserve((fit_mean ? 1 : 0) + 2 * nterms);
        order_indices.reserve((fit_mean ? 1 : 0) + 2 * nterms);
        if (fit_mean) {
            order_types.push_back(static_cast<int>(TermTypeCUDA::Cosine));
            order_indices.push_back(0);
        }
        for (size_t i = 1; i <= nterms; ++i) {
            order_types.push_back(static_cast<int>(TermTypeCUDA::Sine));
            order_indices.push_back(static_cast<int>(i));
            order_types.push_back(static_cast<int>(TermTypeCUDA::Cosine));
            order_indices.push_back(static_cast<int>(i));
        }
        const int k = static_cast<int>(order_types.size());

        int *d_order_types   = nullptr;
        int *d_order_indices = nullptr;
        CUDA_CHECK(cudaMalloc(&d_order_types, order_types.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_order_indices, order_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
           d_order_types,
           order_types.data(),
           order_types.size() * sizeof(int),
           cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
           d_order_indices,
           order_indices.data(),
           order_indices.size() * sizeof(int),
           cudaMemcpyHostToDevice
        ));

        process_chi2_outputs_device_impl(
           d_power,
           d_Sw,
           d_Cw,
           d_Syw,
           d_Cyw,
           d_norm,
           d_order_types,
           d_order_indices,
           k,
           static_cast<int>(nSW),
           static_cast<int>(nSY),
           Nf,
           Nbatch,
           static_cast<int>(norm_kind)
        );

        // Copy result back to host
        Scalar *host_power = new Scalar[Nbatch * Nf];
        nb::capsule owner(host_power, [](void *p) noexcept { delete[] (Scalar *) p; });
        CUDA_CHECK(cudaMemcpy(
           host_power, d_power, Nbatch * Nf * sizeof(Scalar), cudaMemcpyDeviceToHost
        ));
        size_t shape[2] = {Nbatch, Nf};
        auto out =
           nb::ndarray<Scalar, nb::ndim<2>, nb::numpy>(host_power, 2, shape, owner);

        // Cleanup
        cudaFree(d_t);
        cudaFree(d_y);
        cudaFree(d_dy);
        cudaFree(d_t1);
        cudaFree(d_yw);
        cudaFree(d_w);
        cudaFree(d_w2s);
        cudaFree(d_norm);
        cudaFree(d_yws);
        cudaFree(d_Sw);
        cudaFree(d_Cw);
        cudaFree(d_Syw);
        cudaFree(d_Cyw);
        cudaFree(d_tn);
        cudaFree(d_power);
        cudaFree(d_yw_w_work);
        cudaFree(d_fk);
        cudaFree(d_order_types);
        cudaFree(d_order_indices);

        return out;
    }

}  // namespace

NB_MODULE(chi2_cuda_helper, m) {
    m.def(
       "lombscargle_chi2_cuda",
       &lombscargle_chi2_cuda<double>,
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "center_data"_a,
       "fit_mean"_a,
       "norm_kind"_a,
       "nterms"_a     = 1,
       "eps"_a        = 1e-9,
       "gpu_method"_a = 1,
       "block_dim"_a  = -1
    );
    m.def(
       "lombscargle_chi2_cuda",
       &lombscargle_chi2_cuda<float>,
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "center_data"_a,
       "fit_mean"_a,
       "norm_kind"_a,
       "nterms"_a     = 1,
       "eps"_a        = 1e-5f,
       "gpu_method"_a = 1,
       "block_dim"_a  = -1
    );
}
