#ifndef CPU_HELPERS_HPP
#define CPU_HELPERS_HPP

namespace cpu_helpers {
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
}  // namespace cpu_helpers

#endif  // CPU_HELPERS_HPP
