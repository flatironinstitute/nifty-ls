# Changelog

## v1.0.1 (2024-09-12)
Minor optimizations and fixes that make use of finufft v2.3. This version was used in the submitted research note.

OpenMP in the C++ helpers on MacOS ARM has also been fixed, which should result in a small performance improvement for users on M1/M2/etc CPUs.

The finufft version requirement is >= 2.3.

## v1.0.0 (2024-05-28)
Initial release, with support for CPU, GPU, and batched periodograms

The finufft version requirement is >= 2.2.
