#pragma once
#include <complex>
#include <finufft.h>
#include <type_traits>

template <typename Scalar>
struct finufft_plan_type {
    using type =
       std::conditional_t<std::is_same_v<Scalar, float>, finufftf_plan, finufft_plan>;
};

template <typename Scalar>
int _finufft_makeplan(
   int type,
   int dim,
   int64_t *nmodes,
   int isign,
   int ntrans,
   double eps,
   finufft_plan *plan,
   finufft_opts *opts
) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return finufftf_makeplan(
           type,
           dim,
           nmodes,
           isign,
           ntrans,
           eps,
           reinterpret_cast<finufftf_plan *>(plan),
           opts
        );
    } else {
        return finufft_makeplan(type, dim, nmodes, isign, ntrans, eps, plan, opts);
    }
}

template <typename Scalar>
int _finufft_setpts(
   finufft_plan plan,
   size_t M,
   Scalar *xj,
   Scalar *yj,
   Scalar *zj,
   size_t N,
   Scalar *s,
   Scalar *t,
   Scalar *u
) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return finufftf_setpts(
           reinterpret_cast<finufftf_plan>(plan), M, xj, yj, zj, N, s, t, u
        );
    } else {
        return finufft_setpts(plan, M, xj, yj, zj, N, s, t, u);
    }
}

template <typename Scalar>
int _finufft_execute(
   finufft_plan plan, std::complex<Scalar> *weights, std::complex<Scalar> *result
) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return finufftf_execute(reinterpret_cast<finufftf_plan>(plan), weights, result);
    } else {
        return finufft_execute(plan, weights, result);
    }
}

template <typename Scalar>
int _finufft_destroy(finufft_plan plan) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return finufftf_destroy(reinterpret_cast<finufftf_plan>(plan));
    } else {
        return finufft_destroy(plan);
    }
}
