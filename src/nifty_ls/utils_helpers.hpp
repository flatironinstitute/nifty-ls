#ifndef UTILS_HELPERS_HPP
#define UTILS_HELPERS_HPP

#include <algorithm>
#include <vector>

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

namespace utils_helpers {
    enum class NormKind {
        Standard,
        Model,
        Log,
        PSD
    };

    enum class TermType {
        Sine,
        Cosine
    };
}  // namespace utils_helpers

#endif  // UTILS_HELPERS_HPP
