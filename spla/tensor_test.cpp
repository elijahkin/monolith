#include "tensor.hpp"

#include <cmath>
#include <numbers>

#include "gtest/gtest.h"

TEST(TensorTest, VectorAbs) {
  auto a = spla::Tensor<int, 3>::zeros();
  a[0] = -3;
  a[1] = 4;
  a[2] = -2;

  auto b = spla::abs(a);
  EXPECT_EQ(b[0], 3);
  EXPECT_EQ(b[1], 4);
  EXPECT_EQ(b[2], 2);

  EXPECT_EQ(a[0], -3);
  EXPECT_EQ(a[1], 4);
  EXPECT_EQ(a[2], -2);
}

TEST(TensorTest, VectorAdd) {
  auto a = spla::Tensor<int, 4>::ones();
  a[0] = 0;
  a[2] = 0;
  EXPECT_EQ(a.sparsity(), 2);

  auto b = spla::Tensor<int, 4>::full(2);
  b[0] = 0;
  b[1] = 0;
  EXPECT_EQ(b.sparsity(), 2);

  auto c = a + b;
  EXPECT_EQ(c[0], 0);
  EXPECT_EQ(c[1], 1);
  EXPECT_EQ(c[2], 2);
  EXPECT_EQ(c[3], 3);
  EXPECT_EQ(c.sparsity(), 3);

  c += 2;
  EXPECT_EQ(c[0], 2);
  EXPECT_EQ(c[1], 3);
  EXPECT_EQ(c[2], 4);
  EXPECT_EQ(c[3], 5);
  EXPECT_EQ(c.sparsity(), 3);
}

TEST(TensorTest, VectorAll) {
  auto a = spla::Tensor<int, 3>::ones();
  const auto one = a;
  const auto two = spla::Tensor<int, 3>::full(2);
  EXPECT_TRUE(spla::all(a == one));
  EXPECT_FALSE(spla::all(a == two));

  a *= 2;
  EXPECT_FALSE(spla::all(a == one));
  EXPECT_TRUE(spla::all(a == two));

  a -= one;
  EXPECT_TRUE(spla::all(a == one));
  EXPECT_FALSE(spla::all(a == two));
}

TEST(TensorTest, VectorConversion) {
  const auto one_int = spla::Tensor<int, 3>::ones();
  const auto one_double = spla::Tensor<double, 3>::ones();
  const auto two_double = spla::Tensor<double, 3>::full(2);

  auto a = static_cast<spla::Tensor<double, 3>>(one_int);
  EXPECT_TRUE(spla::all(a == one_double));
  EXPECT_FALSE(spla::all(a == two_double));
}

TEST(TensorTest, VectorDot) {
  const auto one = spla::Tensor<int, 5>::ones();
  EXPECT_EQ(spla::dot(one, one), 5);
}

TEST(TensorTest, VectorEq) {
  const auto zero = spla::Tensor<int, 5>::zeros();
  const auto one = spla::Tensor<int, 5>::ones();
  auto a = spla::Tensor<int, 5>::zeros();

  EXPECT_TRUE(spla::all(zero == a));
  EXPECT_TRUE(!spla::any(zero == one));
  EXPECT_TRUE(!spla::all(a == one));

  a[0] = 1;
  EXPECT_TRUE(spla::any(a == one));
}

TEST(TensorTest, VectorExp) {
  auto a = spla::Tensor<double, 2>::zeros();
  a[1] = 1;

  auto exp_v = spla::exp(a);
  EXPECT_NEAR(exp_v[0], 1, 1e-6);
  EXPECT_NEAR(exp_v[1], std::numbers::e_v<double>, 1e-6);

  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
}

TEST(TensorTest, VectorNorm) {
  const auto one = spla::Tensor<double, 3>::ones();
  const auto two = spla::Tensor<double, 3>::full(2);
  const auto three = spla::Tensor<double, 3>::full(3);

  auto a = one;
  EXPECT_NEAR(a[0], 1, 1e-6);
  EXPECT_NEAR(spla::norm(a, 2), std::numbers::sqrt3, 1e-6);

  auto b = two;
  EXPECT_NEAR(b[0], 2, 1e-6);
  EXPECT_NEAR(spla::norm(b, 2), std::sqrt(12), 1e-6);

  a = a + b;
  EXPECT_NEAR(a[0], 3, 1e-6);
  EXPECT_NEAR(spla::norm(a, 2), std::sqrt(27), 1e-6);
  EXPECT_TRUE(spla::all(a == three));
}

TEST(TensorTest, VectorPow) {
  /* const */ auto a = spla::Tensor<int, 4>::full(2);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  auto b = /* spla:: */ pow(a, a);
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1], 4);
  EXPECT_EQ(b[2], 27);
  EXPECT_EQ(b[3], 4);

  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(a[2], 3);
  EXPECT_EQ(a[3], 2);
}

TEST(TensorTest, VectorSum) {
  auto a = spla::Tensor<int, 5>::ones();
  a[2] = 7;
  a[3] = -1;
  EXPECT_EQ(spla::sum(a), 9);
}

TEST(TensorTest, VectorSubscript) {
  auto a = spla::Tensor<int, 10>::zeros();
  EXPECT_NEAR(spla::norm(a, 2), 0, 1e-6);
  EXPECT_EQ(a.sparsity(), 0);

  a[1] = -7;
  a[3] = 4;
  a *= 2;
  EXPECT_EQ(a[1], -14);
  EXPECT_EQ(a[3], 8);
  EXPECT_EQ(a.sparsity(), 2);

  auto b = spla::Tensor<int, 10>::zeros();
  b[3] = 2;
  b[5] = 1;
  EXPECT_EQ(b[3], 2);
  EXPECT_EQ(b[5], 1);
  EXPECT_EQ(b.sparsity(), 2);

  a += b;
  EXPECT_EQ(a[1], -14);
  EXPECT_EQ(a[3], 10);
  EXPECT_EQ(a[5], 1);
  EXPECT_EQ(a.sparsity(), 3);

  EXPECT_NEAR(spla::norm(a, 1), 25, 1e-6);
  EXPECT_NEAR(spla::norm(a, 2), std::sqrt(297), 1e-6);
  EXPECT_EQ(spla::dot(a, b), 21);
}

TEST(TensorTest, MatrixDot) {
  const auto one = spla::Tensor<double, 2, 3>::ones();
  const auto two = spla::Tensor<double, 3, 4>::full(2);
  const auto six = spla::Tensor<double, 2, 4>::full(6);

  // auto a = spla::dot(one, two);
  // EXPECT_TRUE(spla::all(a == six));
}

TEST(TensorTest, MatrixSum) {
  auto a = spla::Tensor<int, 5, 5>::ones();
  const int b = spla::sum(a);
  EXPECT_EQ(b, 25);
}

TEST(TensorTest, MatrixSubscript) {
  auto a = spla::Tensor<float, 2, 2>::zeros();
  a[0, 0] = 1;
  a[1, 1] = 1;
  EXPECT_EQ(a.sparsity(), 2);

  auto b = spla::Tensor<float, 2, 2>::ones();
  b += a;
  // The gtest EXPECT_EQ and EXPECT_NE macros get confused by C++23
  // multidimensional indexing. Hence, we need parentheses around the
  // multidimensional indexed expression for now...
  EXPECT_EQ((b[0, 0]), 2);
  EXPECT_EQ((b[0, 1]), 1);
  EXPECT_EQ((b[1, 0]), 1);
  EXPECT_EQ((b[1, 1]), 2);
  EXPECT_EQ(b.sparsity(), 2);
}

// TODO This test is failing
TEST(TensorTest, MatrixSubscriptFail) {
  auto one = spla::Tensor<int, 3, 3>::ones();
  EXPECT_NE((one[-1, -1]), 1);
  EXPECT_NE((one[5, 7]), 1);
  EXPECT_NE((one[1, 9]), 1);
}

TEST(TensorTest, ScalarAdd) {
  const auto two = spla::Tensor<float>::full(2);
  const auto three = spla::Tensor<float>::full(3);
  // TODO This fails to compile
  // auto a = two + three;
  // EXPECT_EQ(spla::all(a == 5.0F));
}
