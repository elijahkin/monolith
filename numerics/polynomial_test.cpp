#include "polynomial.hpp"

#include "gtest/gtest.h"

TEST(PolynomialTest, Arithmetic) {
  auto x = numerics::Polynomial<int>({0, 1});
  EXPECT_EQ(x.to_string(), "x");

  auto x2 = x;
  x2 *= x;
  EXPECT_EQ(x2.to_string(), "x^2");

  auto p = x2 + x;
  EXPECT_EQ(p.to_string(), "x^2+x");

  auto q = p.deriv();
  EXPECT_EQ(q.to_string(), "2x+1");
}

TEST(PolynomialTest, Comparison) {
  auto p = numerics::Polynomial<int>({3, 0});
  auto q = numerics::Polynomial<int>({3});
  EXPECT_EQ(p, q);
}
