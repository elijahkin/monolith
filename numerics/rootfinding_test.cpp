#include "rootfinding.hpp"

#include <cmath>

#include "gtest/gtest.h"

TEST(RootfindingTest, SquareRoot2) {
  // This function has roots of plus and minus square root 2.
  auto f = [](double x) { return (x * x) - 2; };
  auto df = [](double x) { return 2 * x; };

  double root = 0;
  for (auto bracket : rootfinding::bisection(f, 0.0, 3.0)) {
    root = bracket.midpoint();
  }
  EXPECT_NEAR(root, std::sqrt(2), 1e-10);

  root = 0;
  for (auto x : rootfinding::newton(f, 1.0, df)) {
    root = x;
  }
  EXPECT_NEAR(root, std::sqrt(2), 1e-10);
}

TEST(RootfindingTest, CosineFixedPoint) {
  // Cosine has a unique real fixed point.
  auto f = [](double x) { return cos(x) - x; };
  auto df = [](double x) { return -sin(x) - 1; };

  // https://en.wikipedia.org/wiki/Dottie_number
  constexpr double kDottieNumber = 0.739085133215160641655312087673;

  double root = 0;
  for (auto bracket : rootfinding::bisection(f, 0.0, 3.0)) {
    root = bracket.midpoint();
  }
  EXPECT_NEAR(root, kDottieNumber, 1e-10);

  root = 0;
  for (auto x : rootfinding::newton(f, 1.0, df)) {
    root = x;
  }
  EXPECT_NEAR(root, kDottieNumber, 1e-10);
}
