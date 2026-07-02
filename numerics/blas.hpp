#include <array>
#include <cstddef>
#include <string>

namespace blas {

template <typename T, size_t n, size_t m>
class Matrix {
 public:
  Matrix() = default;

  // Multi-dimensional indexing with C++23
  T& operator[](size_t row, size_t col) { return data_[(row * m) + col]; }

  const T& operator[](size_t row, size_t col) const {
    return data_[(row * m) + col];
  }

  // TODO Is there a more elegant way to do this?
  [[nodiscard]] std::string to_string() const {
    std::string result;
    for (size_t i = 0; i < data_.size(); ++i) {
      result += std::to_string(data_[i]);
      result += ((i - 1) % m == 0) ? '\n' : ' ';
    }
    return result;
  }

 private:
  std::array<T, n * m> data_{};
};

template <typename T, size_t n, size_t m, size_t p>
[[nodiscard]] Matrix<T, n, p> matmul(const Matrix<T, n, m>& lhs,
                                     const Matrix<T, m, p>& rhs) {
  Matrix<T, n, p> result;

  // TODO Can we improve caching with blocking?
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < m; ++k) {
      for (size_t j = 0; j < p; ++j) {
        result[i, j] += lhs[i, k] * rhs[k, j];
      }
    }
  }
  return result;
}

// Matrix power via binary exponentiation
// TODO This should be able to use the monoid power? This would probably involve
// adding a separate matrix identity function.
template <typename T, size_t n>
[[nodiscard]] Matrix<T, n, n> pow(Matrix<T, n, n>& base, size_t exp) {
  Matrix<T, n, n> result;
  result[0, 0] = 1;
  result[1, 1] = 1;

  while (exp > 0) {
    if (exp & 1) {
      result = matmul(result, base);
    }
    base = matmul(base, base);
    exp >>= 1;
  }
  return result;
}

}  // namespace blas
