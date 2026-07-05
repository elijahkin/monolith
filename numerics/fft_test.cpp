#include "fft.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <vector>

#include "gtest/gtest.h"

// Abhinav's DFT for comparison
void dft(std::vector<double>& x, std::vector<std::complex<double>>& output) {
  const size_t N = x.size();
  double theta;

  // Allocate memory for the output array
  output.resize(N, std::complex<double>(0, 0));

  for (size_t k = 0; k < N; ++k) {
    std::complex<double> sum(0.0, 0.0);
    for (size_t n = 0; n < N; ++n) {
      theta = 2 * M_PI * k * n / N;
      std::complex<double> c(std::cos(theta), -std::sin(theta));
      sum += x[n] * c;
    }
    output[k] = sum;
  }
}

TEST(FftTest, DftCompare) {
  const size_t N = 4096;

  std::vector<double> x1(N);
  std::vector<std::complex<double>> x2(N);

  for (size_t i = 0; i < N; ++i) {
    double r = rand();
    x1[i] = r;
    x2[i] = r;
  }

  std::vector<std::complex<double>> chi1(N);
  std::vector<std::complex<double>> chi2(N);

  dft(x1, chi1);
  fft::fft(std::span(x2));

  // Ensure small relative error
  for (size_t i = 0; i < N; ++i) {
    EXPECT_TRUE(abs(chi1[i] - x2[i]) / abs(chi1[i]) < 0.000000001);
  }
}

TEST(FftTest, BenchmarkFft) {
  const size_t N = 2097152;

  std::vector<std::complex<double>> x(N);
  for (size_t i = 0; i < N; ++i) {
    x[i] = rand();
  }

  std::vector<std::complex<double>> chi(N);

  fft::fft(std::span(x));

  // TODO It might be good to time both and expect FFT is at least x times
  // faster than DFT...
}
