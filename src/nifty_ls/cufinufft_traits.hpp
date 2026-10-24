// Shared CUFINUFFT trait wrappers used by multiple CUDA extension modules.
#pragma once

#include <cufinufft.h>

// Type 1 NUFFT
#ifndef CUFINUFFT_TYPE1
#define CUFINUFFT_TYPE1 1
#endif

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
