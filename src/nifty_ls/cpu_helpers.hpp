#ifndef CPU_HELPERS_HPP
#define CPU_HELPERS_HPP

namespace cpu_helpers
{
    enum class NormKind
    {
        Standard,
        Model,
        Log,
        PSD
    };

    enum class TermType
    {
        Sine,
        Cosine
    };
}

#endif // CPU_HELPERS_HPP