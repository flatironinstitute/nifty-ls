# Changelog

## v1.1.0 (2025-07-29)

This release adds `nterms > 1` support, implemented in the `finufft_chi2` and `cufinufft_chi2` backends.  Thanks to @YuWei-CH for the feature!

### Enhancements
- lombscargle: check that all inputs have the same dtype (#59)
- lombscargle: add support for `nterms > 1` (#60)

## v1.0.1 (2024-09-12)
Minor optimizations and fixes that make use of finufft v2.3. This version was used in the submitted research note.

OpenMP in the C++ helpers on MacOS ARM has also been fixed, which should result in a small performance improvement for users on M1/M2/etc CPUs.

The finufft version requirement is >= 2.3.

## v1.0.0 (2024-05-28)
Initial release, with support for CPU, GPU, and batched periodograms

The finufft version requirement is >= 2.2.
