#include "blas.hpp"

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

template <typename T>
T fib_matrix(size_t n) {
  // TODO Cleaner initialization?
  blas::Matrix<T, 2, 2> fib_mat;
  fib_mat[0, 0] = 1;
  fib_mat[0, 1] = 1;
  fib_mat[1, 0] = 1;

  blas::Matrix<T, 2, 1> fib_vec;
  fib_vec[0, 0] = 1;

  auto tmp = pow(fib_mat, n);
  fib_vec = matmul(tmp, fib_vec);
  return fib_vec[1, 0];
}

TEST(BlasTest, Fibonacci) {
  std::vector<int> fib = {0, 1, 1, 2, 3, 5, 8, 13};

  for (size_t i = 0; i < fib.size(); ++i) {
    EXPECT_EQ(fib_matrix<int>(i), fib[i]);
  }
}
