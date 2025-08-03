#ifndef UTILS_HELPERS_HPP
#define UTILS_HELPERS_HPP

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
