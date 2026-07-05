#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

// TODO This file should probably have a corresponding fft.cpp so the user is
// not responsible for adding OpenMP as a dependency.

namespace fft {

// TODO Can anything be made more efficient? SIMD/CUDA potential anywhere?

// ================================== Rings ===================================
// A ring is a set with addition, subtraction, multiplication, additive inverse,
// and distinguished elements 0 (additive identity) and 1 (multiplicative
// identity). Note: concepts enforce syntax only.

// Rings are the natural algebraic setting for the elements of discrete Fourier
// transform. The classic FFT would use the ring of complex numbers, but the
// number theoretic transform is done over some finite field.

template <class T>
concept Ring = requires(T a, T b) {
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { a *b } -> std::same_as<T>;
  T{0};
  T{1};
};

// ========================== Fast Fourier Transform ==========================

// In-place radix-2 Cooley-Tukey algorithm with general rings
// TODO Maybe we want radix-4 in the long term for better performance?
template <Ring T>
class FFTPlan {
 public:
  explicit FFTPlan(size_t N, bool inverse, T primitive_root)
      : N_(N), inverse_(inverse) {
    assert((N & (N - 1)) == 0 && "Radix-2 FFT requires N to be a power of two");

    // First construct a flat twiddle table from the primitive root
    std::vector<T> twiddle_table(N / 2);
    T twiddle = T{1};
    for (size_t i = 0; i < N / 2; ++i) {
      twiddle_table[i] = twiddle;
      twiddle *= primitive_root;
    }

    // Convert to a stage-indexed table for better cache locality
    for (size_t m = 2; m <= N_; m *= 2) {
      const size_t half = m / 2;
      const size_t step = N_ / m;
      auto &stage = twiddle_stages_.emplace_back(half);
      for (size_t k = 0; k < half; ++k) {
        stage[k] = twiddle_table[k * step];
      }
    }
  }

  // TODO Does this function need to exist? Or just make the other functions
  // public?
  void execute(std::span<T> x) {
    assert(x.size() == N_ && "Input size must match plan size");
    bit_reverse(x);
    butterfly(x);
    if (inverse_) {
      normalize(x);
    }
  }

 private:
  std::vector<std::vector<T>> twiddle_stages_;
  size_t N_;
  bool inverse_;

  void bit_reverse(std::span<T> x) {
    // TODO Precompute the swaps to save when doing multiple computations?
    size_t j = 0;
    for (size_t k = 1; k < N_; ++k) {
      size_t bit = N_ >> 1U;
      for (; j & bit; bit >>= 1U) {
        j ^= bit;
      }
      j ^= bit;
      if (k < j) {
        std::swap(x[k], x[j]);
      }
    }
  }

  void butterfly(std::span<T> x) {
#pragma omp parallel
    {
      for (size_t m = 2, stage = 0; m <= N_; m *= 2, ++stage) {
        const size_t half = m / 2;

#pragma omp for schedule(static)
        for (size_t j = 0; j < N_; j += m) {
          // TODO Can we give a SIMD hint here?
          for (size_t k = 0; k < half; ++k) {
            const T omega = twiddle_stages_[stage][k];

            const T p = x[j + k];
            const T q = x[j + k + half] * omega;

            x[j + k] = p + q;
            x[j + k + half] = p - q;
          }
        }
      }
    }
  }

  // TODO Parallelize this?, And again, make it static?
  // TODO Doesn't this technically need a field instead of a ring?
  void normalize(std::span<T> x) {
    const T N_inv = T{1} / T(N_);
    for (auto &v : x) {
      v *= N_inv;
    }
  }
};

template <typename T>
void fft(std::span<std::complex<T>> x) {
  // TODO If x isn't a power of 2, we can pad it. Do we "unpad" after the
  // transform?

  const T angle = -2 * std::numbers::pi_v<T> / x.size();
  const std::complex<T> root = std::polar(T{1}, angle);

  auto plan = FFTPlan(x.size(), false, root);
  plan.execute(x);
}

template <typename T>
void ifft(std::span<std::complex<T>> x) {
  // TODO If x isn't a power of 2, we can pad it. Do we "unpad" after the
  // transform?

  const T angle = 2 * std::numbers::pi_v<T> / x.size();
  const std::complex<T> root = std::polar(T{1}, angle);

  auto plan = FFTPlan(x.size(), true, root);
  plan.execute(x);
}

}  // namespace fft
