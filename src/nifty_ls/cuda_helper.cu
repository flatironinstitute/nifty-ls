// CUDA implementation of Lomb-Scargle helpers and cufinufft bindings.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/complex.h>

#include <cufinufft.h>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include "utils_helpers.hpp"
#include "cufinufft_traits.hpp"
using utils_helpers::NormKind;

namespace nb = nanobind;
using namespace nb::literals;

namespace {

    constexpr int TPB = 256;

    template <typename Scalar>
    using Complex = thrust::complex<Scalar>;

    template <typename Scalar>
    using nifty_cuda_arr_1d = nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda>;

    template <typename Scalar>
    using nifty_cuda_arr_2d = nb::ndarray<Scalar, nb::ndim<2>, nb::device::cuda>;

    // Define post-process prototype
    template <typename Scalar>
    void process_finufft_outputs_cuda(
       nifty_cuda_arr_2d<Scalar> power_,
       nifty_cuda_arr_2d<const Complex<Scalar>> f1_,
       nifty_cuda_arr_2d<const Complex<Scalar>> fw_,
       nifty_cuda_arr_2d<const Complex<Scalar>> f2_,
       nifty_cuda_arr_1d<const Scalar> norm_YY_,
       const NormKind norm_kind,
       const bool fit_mean,
       int block_dim
    );

#define CUDA_CHECK(expr)                                        \
    do {                                                        \
        cudaError_t _err = (expr);                              \
        if (_err != cudaSuccess) {                              \
            throw std::runtime_error(cudaGetErrorString(_err)); \
        }                                                       \
    } while (0)

    // partial_wsum_yoff kernel: For each batch, calculate the weight w = 1/dy^2 for dy
    // and write it into w2. Then, perform block reduction to obtain wsum_partial and
    // yoff_partial.
    template <typename Scalar>
    __global__ void partial_wsum_yoff(
       const Scalar *dy,
       const Scalar *y,
       Complex<Scalar> *w2,
       double *wsum_partial,
       double *yoff_partial,
       const bool accumulate_yoff,
       const size_t N,
       const size_t blocks_per_batch
    ) {
        const size_t j     = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t batch = blockIdx.y;

        // Shared reductions for this block: first half wsum, second half yoff
        extern __shared__ double scratch[];
        double *s_wsum = scratch;
        double *s_yoff = scratch + blockDim.x;

        double wsum_val = 0.0;
        double yoff_val = 0.0;

        if (j < N) {
            const size_t idx    = batch * N + j;
            const Scalar dy_val = dy[idx];
            const Scalar w_val  = Scalar(1) / (dy_val * dy_val);
            w2[idx]             = Complex<Scalar>(w_val, Scalar(0));
            wsum_val            = static_cast<double>(w_val);
            if (accumulate_yoff) { yoff_val = static_cast<double>(w_val * y[idx]); }
        }

        s_wsum[threadIdx.x] = wsum_val;
        s_yoff[threadIdx.x] = yoff_val;
        __syncthreads();

        // Block reduction
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_wsum[threadIdx.x] += s_wsum[threadIdx.x + stride];
                s_yoff[threadIdx.x] += s_yoff[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Write final result
        if (threadIdx.x == 0) {
            const size_t block_idx  = batch * blocks_per_batch + blockIdx.x;
            wsum_partial[block_idx] = s_wsum[0];
            yoff_partial[block_idx] = s_yoff[0];
        }
    }

    // reduce_partials2 kernel: Reduce two partial arrays in one pass.
    __global__ void reduce_partials2(
       const double *partial1,
       const double *partial2,
       double *total1,
       double *total2,
       const size_t blocks_per_batch
    ) {
        const size_t batch = blockIdx.x;

        extern __shared__ double scratch[];
        double *s1 = scratch;
        double *s2 = scratch + blockDim.x;

        double sum1 = 0.0;
        double sum2 = 0.0;
        for (size_t b = threadIdx.x; b < blocks_per_batch; b += blockDim.x) {
            const size_t idx = batch * blocks_per_batch + b;
            sum1 += partial1[idx];
            sum2 += partial2[idx];
        }
        s1[threadIdx.x] = sum1;
        s2[threadIdx.x] = sum2;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s1[threadIdx.x] += s1[threadIdx.x + stride];
                s2[threadIdx.x] += s2[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            total1[batch] = s1[0];
            total2[batch] = s2[0];
        }
    }

    // set_norm_and_yoff kernel: Execute yoff /= wsum (if center/fit), and set norm (PSD
    // or 0)
    template <typename Scalar>
    __global__ void set_norm_and_yoff(
       const double *wsum,
       double *yoff,
       Scalar *norm,
       const bool center_or_fit,
       const bool psd_norm,
       const size_t Nbatch
    ) {
        const size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= Nbatch) { return; }

        const double wsum_i = wsum[batch];
        if (center_or_fit) { yoff[batch] /= wsum_i; }
        norm[batch] = psd_norm ? static_cast<Scalar>(wsum_i) : Scalar(0);
    }

    // finalize_phase_shift kernel: Calulate t1, t2, normalize w2, construct yw/w/w2,
    // and accumulate norm using shared reduction and atomicAdd (non-PSD cases)
    template <typename Scalar>
    __global__ void finalize_phase_shift(
       const Scalar *t,
       const Scalar *y,
       const double *wsum,
       const double *yoff,
       Complex<Scalar> *w2,
       Complex<Scalar> *yw,
       Complex<Scalar> *w,
       Scalar *t1,
       Scalar *t2,
       Scalar *norm,
       const Scalar fmin,
       const Scalar df,
       const size_t Nf,
       const bool center_data,
       const bool fit_mean,
       const bool psd_norm,
       const size_t N
    ) {
        const size_t j     = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t batch = blockIdx.y;
        if (j >= N) { return; }

        const size_t idx    = batch * N + j;
        const Scalar TWO_PI = static_cast<Scalar>(2 * CUDART_PI);
        const Scalar t1_val = TWO_PI * df * t[j];
        const Scalar t2_val = t1_val + t1_val;

        if (batch == 0) {
            t1[j] = t1_val;
            t2[j] = t2_val;
        }

        const double wsum_i = wsum[batch];
        const Scalar yoff_i =
           (center_data || fit_mean) ? static_cast<Scalar>(yoff[batch]) : Scalar(0);

        const Scalar w2_norm = w2[idx].real() / static_cast<Scalar>(wsum_i);
        w2[idx]              = Complex<Scalar>(w2_norm, Scalar(0));

        // Accumulate norm per block, then atomically add once per block.
        extern __shared__ double s_norm[];
        double partial = 0.0;
        if (!psd_norm) {
            const Scalar diff = y[idx] - yoff_i;
            partial           = static_cast<double>(w2_norm * diff * diff);
        }
        s_norm[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_norm[threadIdx.x] += s_norm[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && !psd_norm) {
            atomicAdd(&norm[batch], static_cast<Scalar>(s_norm[0]));
        }

        const Scalar phase_factor = static_cast<Scalar>(Nf / 2) + fmin / df;
        const Scalar angle        = phase_factor * t1_val;
        Scalar sin_angle          = Scalar(0);
        Scalar cos_angle          = Scalar(1);
        if constexpr (std::is_same_v<Scalar, float>) {
            sincosf(angle, &sin_angle, &cos_angle);
        } else {
            sincos(angle, &sin_angle, &cos_angle);
        }
        const Complex<Scalar> exp_t1(cos_angle, sin_angle);

        const Scalar y_adj           = y[idx] - yoff_i;
        const Complex<Scalar> w2_val = w2[idx];  // normalized: imag == 0

        const Complex<Scalar> w2_exp_t1 = w2_val * exp_t1;
        yw[idx]                         = y_adj * w2_exp_t1;
        if (fit_mean) { w[idx] = w2_exp_t1; }
        w2[idx] = w2_exp_t1 * exp_t1;  // exp(i*2*angle)
    }

    // process_finufft_outputs_kernel kernel: Compute Lomb-Scargle power using
    // f1/fw/f2/norm_YY
    template <typename Scalar>
    __global__ void process_finufft_outputs_kernel(
       Scalar *power,
       const Complex<Scalar> *f1,
       const Complex<Scalar> *fw,
       const Complex<Scalar> *f2,
       const Scalar *norm_YY,
       const NormKind norm_kind,
       const bool fit_mean,
       const size_t N,
       const size_t Nbatch
    ) {
        const size_t j     = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t batch = blockIdx.y;
        if (j >= N) { return; }

        const size_t idx = batch * N + j;

        const Complex<Scalar> f1v = f1[idx];
        const Complex<Scalar> fwv = fw[idx];
        const Complex<Scalar> f2v = f2[idx];

        Scalar tan_2omega_tau;
        if (fit_mean) {
            tan_2omega_tau =
               (f2v.imag() - 2 * fwv.imag() * fwv.real())
               / (f2v.real() - (fwv.real() * fwv.real() - fwv.imag() * fwv.imag()));
        } else {
            tan_2omega_tau = f2v.imag() / f2v.real();
        }

        // scalar operations for register data
        const Scalar S2w =
           tan_2omega_tau / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
        const Scalar C2w = Scalar(1) / std::sqrt(1 + tan_2omega_tau * tan_2omega_tau);
        const Scalar Cw  = static_cast<Scalar>(std::sqrt(0.5)) * std::sqrt(1 + C2w);
        const Scalar Sw  = static_cast<Scalar>(std::sqrt(0.5)) * (S2w >= 0 ? 1 : -1)
                          * std::sqrt(1 - C2w);

        const Scalar YC = f1v.real() * Cw + f1v.imag() * Sw;
        const Scalar YS = f1v.imag() * Cw - f1v.real() * Sw;
        Scalar CC       = (1 + f2v.real() * C2w + f2v.imag() * S2w) / 2;
        Scalar SS       = (1 - f2v.real() * C2w - f2v.imag() * S2w) / 2;

        if (fit_mean) {
            const Scalar CC_fac = fwv.real() * Cw + fwv.imag() * Sw;
            const Scalar SS_fac = fwv.imag() * Cw - fwv.real() * Sw;
            CC -= CC_fac * CC_fac;
            SS -= SS_fac * SS_fac;
        }

        Scalar p = YC * YC / CC + YS * YS / SS;

        switch (norm_kind) {
            case NormKind::Standard:
                p /= norm_YY[batch];
                break;
            case NormKind::Model:
                p /= norm_YY[batch] - p;
                break;
            case NormKind::Log:
                p = -std::log(1 - p / norm_YY[batch]);
                break;
            case NormKind::PSD:
                p *= static_cast<Scalar>(0.5) * norm_YY[batch];
                break;
        }

        power[idx] = p;
    }

    inline int clamp_block_dim_host(int block_dim) {
        if (block_dim <= 0) { return TPB; }
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        return std::min(block_dim, prop.maxThreadsPerBlock);
    }

    inline void CUFINUFFT_CHECK(int code, const char *msg) {
        if (code != 0) { throw std::runtime_error(msg); }
    }

    template <typename Scalar>
    void cufinufft_type1_execute(
       const Scalar *x,
       const Complex<Scalar> *c,
       int64_t M,
       int64_t Nf,
       int64_t ntrans,
       Complex<Scalar> *fk,
       double eps,
       int gpu_method
    ) {
        int iflag         = 1;
        int64_t dim       = 1;
        int64_t nmodes[1] = {Nf};

        using Traits                 = CufinufftTraits<Scalar>;
        typename Traits::plan_t plan = nullptr;
        cufinufft_opts opts;
        cufinufft_default_opts(&opts);
        opts.gpu_method = gpu_method;

        // TODO: DEBUG
        CUFINUFFT_CHECK(
           Traits::makeplan(dim, nmodes, iflag, ntrans, eps, &plan, &opts),
           "cufinufft_makeplan failed"
        );

        CUFINUFFT_CHECK(Traits::setpts(plan, M, x), "cufinufft_setpts failed");

        CUFINUFFT_CHECK(
           Traits::execute(
              plan,
              reinterpret_cast<const typename Traits::complex_t *>(c),
              reinterpret_cast<typename Traits::complex_t *>(fk)
           ),
           "cufinufft_execute failed"
        );

        CUFINUFFT_CHECK(Traits::destroy(plan), "cufinufft_destroy failed");
    }

    template <typename Scalar>
    void process_finufft_inputs_cuda(
       nifty_cuda_arr_1d<Scalar> t1_,
       nifty_cuda_arr_1d<Scalar> t2_,
       nifty_cuda_arr_2d<Complex<Scalar>> yw_,
       nifty_cuda_arr_2d<Complex<Scalar>> w_,
       nifty_cuda_arr_2d<Complex<Scalar>> w2_,
       nifty_cuda_arr_1d<Scalar> norm_,
       nifty_cuda_arr_1d<const Scalar> t_,
       nifty_cuda_arr_2d<const Scalar> y_,
       nifty_cuda_arr_2d<const Scalar> dy_,
       const Scalar fmin,
       const Scalar df,
       const size_t Nf,
       const bool center_data,
       const bool fit_mean,
       const bool psd_norm
    ) {
        Scalar *t1          = t1_.data();
        Scalar *t2          = t2_.data();
        Complex<Scalar> *yw = yw_.data();
        Complex<Scalar> *w  = w_.data();
        Complex<Scalar> *w2 = w2_.data();
        Scalar *norm        = norm_.data();
        const Scalar *t     = t_.data();
        const Scalar *y     = y_.data();
        const Scalar *dy    = dy_.data();

        const size_t Nbatch = y_.shape(0);
        const size_t N      = y_.shape(1);

        const size_t blocks_per_batch = (N + TPB - 1) / TPB;

        double *wsum_partial = nullptr;
        double *yoff_partial = nullptr;
        double *wsum         = nullptr;
        double *yoff         = nullptr;

        CUDA_CHECK(
           cudaMalloc(&wsum_partial, Nbatch * blocks_per_batch * sizeof(double))
        );
        CUDA_CHECK(
           cudaMalloc(&yoff_partial, Nbatch * blocks_per_batch * sizeof(double))
        );
        CUDA_CHECK(cudaMalloc(&wsum, Nbatch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&yoff, Nbatch * sizeof(double)));

        const dim3 grid_partial(blocks_per_batch, Nbatch);
        const size_t shmem_partial = 2 * TPB * sizeof(double);
        partial_wsum_yoff<<<grid_partial, TPB, shmem_partial>>>(
           dy,
           y,
           w2,
           wsum_partial,
           yoff_partial,
           center_data || fit_mean,
           N,
           blocks_per_batch
        );
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_reduce(Nbatch);
        const size_t shmem_reduce = 2 * TPB * sizeof(double);
        reduce_partials2<<<grid_reduce, TPB, shmem_reduce>>>(
           wsum_partial, yoff_partial, wsum, yoff, blocks_per_batch
        );
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_batch((Nbatch + TPB - 1) / TPB);
        set_norm_and_yoff<<<grid_batch, TPB>>>(
           wsum, yoff, norm, center_data || fit_mean, psd_norm, Nbatch
        );
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_finalize(blocks_per_batch, Nbatch);
        const size_t shmem_norm = TPB * sizeof(double);
        finalize_phase_shift<<<grid_finalize, TPB, shmem_norm>>>(
           t,
           y,
           wsum,
           yoff,
           w2,
           yw,
           w,
           t1,
           t2,
           norm,
           fmin,
           df,
           Nf,
           center_data,
           fit_mean,
           psd_norm,
           N
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(wsum_partial));
        CUDA_CHECK(cudaFree(yoff_partial));
        CUDA_CHECK(cudaFree(wsum));
        CUDA_CHECK(cudaFree(yoff));
    }

    // Aliases for cufinufft preprocessing/postprocessing; reuse same kernels
    // Help re-using in heterobatch version
    template <typename Scalar>
    void process_cufinufft_inputs_cuda(
       nifty_cuda_arr_1d<Scalar> t1_,
       nifty_cuda_arr_1d<Scalar> t2_,
       nifty_cuda_arr_2d<Complex<Scalar>> yw_,
       nifty_cuda_arr_2d<Complex<Scalar>> w_,
       nifty_cuda_arr_2d<Complex<Scalar>> w2_,
       nifty_cuda_arr_1d<Scalar> norm_,
       nifty_cuda_arr_1d<const Scalar> t_,
       nifty_cuda_arr_2d<const Scalar> y_,
       nifty_cuda_arr_2d<const Scalar> dy_,
       const Scalar fmin,
       const Scalar df,
       const size_t Nf,
       const bool center_data,
       const bool fit_mean,
       const bool psd_norm
    ) {
        process_finufft_inputs_cuda(
           t1_,
           t2_,
           yw_,
           w_,
           w2_,
           norm_,
           t_,
           y_,
           dy_,
           fmin,
           df,
           Nf,
           center_data,
           fit_mean,
           psd_norm
        );
    }

    template <typename Scalar>
    void process_cufinufft_outputs_cuda(
       nifty_cuda_arr_2d<Scalar> power_,
       nifty_cuda_arr_2d<const Complex<Scalar>> f1_,
       nifty_cuda_arr_2d<const Complex<Scalar>> fw_,
       nifty_cuda_arr_2d<const Complex<Scalar>> f2_,
       nifty_cuda_arr_1d<const Scalar> norm_YY_,
       const NormKind norm_kind,
       const bool fit_mean,
       int block_dim
    ) {
        process_finufft_outputs_cuda(
           power_, f1_, fw_, f2_, norm_YY_, norm_kind, fit_mean, block_dim
        );
    }

    template <typename Scalar>
    void process_finufft_outputs_cuda(
       nifty_cuda_arr_2d<Scalar> power_,
       nifty_cuda_arr_2d<const Complex<Scalar>> f1_,
       nifty_cuda_arr_2d<const Complex<Scalar>> fw_,
       nifty_cuda_arr_2d<const Complex<Scalar>> f2_,
       nifty_cuda_arr_1d<const Scalar> norm_YY_,
       const NormKind norm_kind,
       const bool fit_mean,
       int block_dim
    ) {
        Scalar *power             = power_.data();
        const Complex<Scalar> *f1 = f1_.data();
        const Complex<Scalar> *fw = fw_.data();
        const Complex<Scalar> *f2 = f2_.data();
        const Scalar *norm_YY     = norm_YY_.data();

        const size_t Nbatch = f1_.shape(0);
        const size_t N      = f1_.shape(1);

        const int threads = clamp_block_dim_host(block_dim);
        const dim3 grid((N + threads - 1) / threads, Nbatch);

        process_finufft_outputs_kernel<<<grid, threads>>>(
           power, f1, fw, f2, norm_YY, norm_kind, fit_mean, N, Nbatch
        );
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename Scalar>
    void cufinufft_type1_cuda(
       nifty_cuda_arr_1d<const Scalar> t1_,
       nifty_cuda_arr_2d<const Complex<Scalar>> c_,
       nifty_cuda_arr_2d<Complex<Scalar>> fk_,
       int64_t Nf,
       double eps,
       int gpu_method
    ) {
        const Scalar *t1         = t1_.data();
        const Complex<Scalar> *c = c_.data();
        Complex<Scalar> *fk      = fk_.data();

        const int64_t ntrans = static_cast<int64_t>(c_.shape(0));
        const int64_t M      = static_cast<int64_t>(c_.shape(1));

        cufinufft_type1_execute(t1, c, M, Nf, ntrans, fk, eps, gpu_method);
    }

    // Bind lombscargle_cuda
    template <typename Scalar>
    nb::ndarray<Scalar, nb::ndim<2>, nb::numpy> lombscargle_cuda(
       nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cpu> t_,
       nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cpu> y_,
       nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cpu> dy_,
       const Scalar fmin,
       const Scalar df,
       const size_t Nf,
       const bool center_data,
       const bool fit_mean,
       const NormKind norm_kind,
       const double eps,
       const int gpu_method,
       const int block_dim
    ) {
        // Shapes
        const size_t N       = y_.shape(1);
        const size_t Nbatch  = y_.shape(0);
        const int tpb_launch = clamp_block_dim_host(block_dim);

        // Device allocations
        Scalar *d_t    = nullptr;
        Scalar *d_t1   = nullptr;
        Scalar *d_t2   = nullptr;
        Scalar *d_norm = nullptr;
        Scalar *d_y    = nullptr;
        Scalar *d_dy   = nullptr;

        const size_t NN         = Nbatch * N;
        const size_t yww_trans  = fit_mean ? (2 * Nbatch) : Nbatch;
        const size_t yww_elems  = yww_trans * N;
        const size_t f1fw_elems = yww_trans * Nf;

        Complex<Scalar> *d_yw_w = nullptr;
        Complex<Scalar> *d_w2   = nullptr;
        Complex<Scalar> *d_f1fw = nullptr;
        Complex<Scalar> *d_f2   = nullptr;
        Scalar *d_power         = nullptr;

        double *wsum_partial = nullptr;
        double *yoff_partial = nullptr;
        double *wsum         = nullptr;
        double *yoff         = nullptr;

        CUDA_CHECK(cudaMalloc(&d_t, N * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_t1, N * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_t2, N * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_norm, Nbatch * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_y, NN * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_dy, NN * sizeof(Scalar)));

        CUDA_CHECK(cudaMalloc(&d_yw_w, yww_elems * sizeof(Complex<Scalar>)));
        CUDA_CHECK(cudaMalloc(&d_w2, NN * sizeof(Complex<Scalar>)));
        CUDA_CHECK(cudaMalloc(&d_f1fw, f1fw_elems * sizeof(Complex<Scalar>)));
        CUDA_CHECK(cudaMalloc(&d_f2, Nbatch * Nf * sizeof(Complex<Scalar>)));
        CUDA_CHECK(cudaMalloc(&d_power, Nbatch * Nf * sizeof(Scalar)));

        // Copy inputs
        CUDA_CHECK(
           cudaMemcpy(d_t, t_.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice)
        );
        CUDA_CHECK(
           cudaMemcpy(d_y, y_.data(), NN * sizeof(Scalar), cudaMemcpyHostToDevice)
        );
        CUDA_CHECK(
           cudaMemcpy(d_dy, dy_.data(), NN * sizeof(Scalar), cudaMemcpyHostToDevice)
        );

        const size_t blocks_per_batch = (N + tpb_launch - 1) / tpb_launch;
        CUDA_CHECK(
           cudaMalloc(&wsum_partial, Nbatch * blocks_per_batch * sizeof(double))
        );
        CUDA_CHECK(
           cudaMalloc(&yoff_partial, Nbatch * blocks_per_batch * sizeof(double))
        );
        CUDA_CHECK(cudaMalloc(&wsum, Nbatch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&yoff, Nbatch * sizeof(double)));

        // Preprocess
        const dim3 grid_partial(blocks_per_batch, Nbatch);
        const size_t shmem_partial = 2 * tpb_launch * sizeof(double);
        partial_wsum_yoff<<<grid_partial, tpb_launch, shmem_partial>>>(
           d_dy,
           d_y,
           d_w2,
           wsum_partial,
           yoff_partial,
           center_data || fit_mean,
           N,
           blocks_per_batch
        );
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_reduce(Nbatch);
        const size_t shmem_reduce = 2 * tpb_launch * sizeof(double);
        reduce_partials2<<<grid_reduce, tpb_launch, shmem_reduce>>>(
           wsum_partial, yoff_partial, wsum, yoff, blocks_per_batch
        );
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_batch((Nbatch + tpb_launch - 1) / tpb_launch);
        set_norm_and_yoff<<<grid_batch, tpb_launch>>>(
           wsum,
           yoff,
           d_norm,
           center_data || fit_mean,
           norm_kind == NormKind::PSD,
           Nbatch
        );
        CUDA_CHECK(cudaGetLastError());

        Complex<Scalar> *d_yw = d_yw_w;
        Complex<Scalar> *d_w  = fit_mean ? (d_yw_w + Nbatch * N) : nullptr;

        const dim3 grid_finalize(blocks_per_batch, Nbatch);
        const size_t shmem_norm = tpb_launch * sizeof(double);
        finalize_phase_shift<<<grid_finalize, tpb_launch, shmem_norm>>>(
           d_t,
           d_y,
           wsum,
           yoff,
           d_w2,
           d_yw,
           d_w ? d_w : d_yw,  // w unused if !fit_mean
           d_t1,
           d_t2,
           d_norm,
           fmin,
           df,
           Nf,
           center_data,
           fit_mean,
           norm_kind == NormKind::PSD,
           N
        );
        CUDA_CHECK(cudaGetLastError());

        // NUFFT type-1: first pass (yw_w)
        const int64_t ntrans1 = static_cast<int64_t>(yww_trans);
        cufinufft_type1_execute(
           d_t1,
           d_yw_w,
           static_cast<int64_t>(N),
           static_cast<int64_t>(Nf),
           ntrans1,
           d_f1fw,
           eps,
           gpu_method
        );

        // NUFFT type-1: second pass (w2)
        cufinufft_type1_execute(
           d_t2,
           d_w2,
           static_cast<int64_t>(N),
           static_cast<int64_t>(Nf),
           static_cast<int64_t>(Nbatch),
           d_f2,
           eps,
           gpu_method
        );

        // Postprocess
        const int threads = clamp_block_dim_host(block_dim);
        const dim3 grid_out((Nf + threads - 1) / threads, Nbatch);
        process_finufft_outputs_kernel<<<grid_out, threads>>>(
           d_power,
           d_f1fw,
           fit_mean ? d_f1fw + Nbatch * Nf : d_f1fw,  // fw only valid if fit_mean
           d_f2,
           d_norm,
           norm_kind,
           fit_mean,
           Nf,
           Nbatch
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy back
        size_t shape[2]  = {Nbatch, Nf};
        Scalar *host_ptr = new Scalar[Nbatch * Nf];
        nb::capsule owner(host_ptr, [](void *p) noexcept { delete[] (Scalar *) p; });

        CUDA_CHECK(cudaMemcpy(
           host_ptr, d_power, Nbatch * Nf * sizeof(Scalar), cudaMemcpyDeviceToHost
        ));

        auto power_out =
           nb::ndarray<Scalar, nb::ndim<2>, nb::numpy>(host_ptr, 2, shape, owner);

        // Free
        cudaFree(d_t);
        cudaFree(d_t1);
        cudaFree(d_t2);
        cudaFree(d_norm);
        cudaFree(d_y);
        cudaFree(d_dy);
        cudaFree(d_yw_w);
        cudaFree(d_w2);
        cudaFree(d_f1fw);
        cudaFree(d_f2);
        cudaFree(d_power);
        cudaFree(wsum_partial);
        cudaFree(yoff_partial);
        cudaFree(wsum);
        cudaFree(yoff);

        return power_out;
    }

}  // namespace

NB_MODULE(cuda_helper, m) {
    // noconvert to avoid host staging; requires device arrays
    m.def(
       "process_finufft_inputs",
       &process_finufft_inputs_cuda<double>,
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
       "psd_normalization"_a
    );

    m.def(
       "process_finufft_inputs",
       &process_finufft_inputs_cuda<float>,
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
       "psd_normalization"_a
    );

    m.def(
       "process_finufft_outputs",
       &process_finufft_outputs_cuda<double>,
       "power"_a.noconvert(),
       "f1"_a.noconvert(),
       "fw"_a.noconvert(),
       "f2"_a.noconvert(),
       "norm_YY"_a.noconvert(),
       "norm_kind"_a,
       "fit_mean"_a,
       "block_dim"_a = -1
    );

    m.def(
       "process_finufft_outputs",
       &process_finufft_outputs_cuda<float>,
       "power"_a.noconvert(),
       "f1"_a.noconvert(),
       "fw"_a.noconvert(),
       "f2"_a.noconvert(),
       "norm_YY"_a.noconvert(),
       "norm_kind"_a,
       "fit_mean"_a,
       "block_dim"_a = -1
    );

    // cufinufft type1 wrapper (GPU) using cufinufft C API
    m.def(
       "cufinufft_type1",
       &cufinufft_type1_cuda<double>,
       "t1"_a.noconvert(),
       "c"_a.noconvert(),
       "fk"_a.noconvert(),
       "Nf"_a,
       "eps"_a        = 1e-9,
       "gpu_method"_a = 1
    );

    m.def(
       "cufinufft_type1",
       &cufinufft_type1_cuda<float>,
       "t1"_a.noconvert(),
       "c"_a.noconvert(),
       "fk"_a.noconvert(),
       "Nf"_a,
       "eps"_a        = 1e-5f,
       "gpu_method"_a = 1
    );

    // cufinufft helpers reuse the same CUDA kernels
    m.def(
       "process_cufinufft_inputs",
       &process_cufinufft_inputs_cuda<double>,
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
       "psd_normalization"_a
    );

    m.def(
       "process_cufinufft_inputs",
       &process_cufinufft_inputs_cuda<float>,
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
       "psd_normalization"_a
    );

    m.def(
       "process_cufinufft_outputs",
       &process_cufinufft_outputs_cuda<double>,
       "power"_a.noconvert(),
       "f1"_a.noconvert(),
       "fw"_a.noconvert(),
       "f2"_a.noconvert(),
       "norm_YY"_a.noconvert(),
       "norm_kind"_a,
       "fit_mean"_a,
       "block_dim"_a = -1
    );

    m.def(
       "process_cufinufft_outputs",
       &process_cufinufft_outputs_cuda<float>,
       "power"_a.noconvert(),
       "f1"_a.noconvert(),
       "fw"_a.noconvert(),
       "f2"_a.noconvert(),
       "norm_YY"_a.noconvert(),
       "norm_kind"_a,
       "fit_mean"_a,
       "block_dim"_a = -1
    );

    // End-to-end path without CuPy: host arrays in/out
    m.def(
       "lombscargle_cuda",
       &lombscargle_cuda<double>,
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "center_data"_a,
       "fit_mean"_a,
       "norm_kind"_a,
       "eps"_a        = 1e-9,
       "gpu_method"_a = 1,
       "block_dim"_a  = -1
    );

    m.def(
       "lombscargle_cuda",
       &lombscargle_cuda<float>,
       "t"_a.noconvert(),
       "y"_a.noconvert(),
       "dy"_a.noconvert(),
       "fmin"_a,
       "df"_a,
       "Nf"_a,
       "center_data"_a,
       "fit_mean"_a,
       "norm_kind"_a,
       "eps"_a        = 1e-5f,
       "gpu_method"_a = 1,
       "block_dim"_a  = -1
    );
}
