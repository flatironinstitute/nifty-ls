#include "finufft.h"
#include <vector>
#include <complex>
#include <stdlib.h>
#include <iostream>

#include <nanobind/nanobind.h>

void test()
{
    int M = 1e3; // number of nonuniform points
    std::vector<double> x(M);
    std::vector<std::complex<double>> c(M);
    std::complex<double> I = std::complex<double>(0.0, 1.0); // the imaginary unit
    for (int j = 0; j < M; ++j)
    {
        x[j] = M_PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
        c[j] = 2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
    }

    int N = 1e2; // number of output modes
    std::vector<std::complex<double>> F(N);

    int ier = finufft1d1(M, &x[0], &c[0], +1, 1e-9, N, &F[0], NULL);

    // print result
    if (ier == 0) {
        for (int j = 0; j < N; ++j) {
            std::cout << "F[" << j << "] = " << F[j] << std::endl;
        }
    } else {
        std::cerr << "Error in finufft1d1: " << ier << std::endl;
    }
}

// nanobind interface
NB_MODULE(heterobatch_helpers, m) {
    m.def("test", &test, "Test function for finufft1d1");
}
