#include "ode.hpp"

#include <cmath>

#include "gtest/gtest.h"

TEST(OdeTest, FirstOrderLinear) {
  auto f = [](double t, double y) { return (t * t * t) + y; };

  // Exact solution found via integrating factors and three applications of
  // integration by parts. Yuck!
  auto y_exact = [](double t) {
    return -(t * t * t) - (3 * t * t) - (6 * t) - 6 + (5 * std::exp(t));
  };

  auto params = ode::IntegrationParams<double, double>{
      .t0 = 0, .y0 = -1, .h = 1e-5, .num_iters = 300000};

  for (auto const& [t, y] : solve_ivp<ode::methods::kRK4<double>>(f, params)) {
    EXPECT_NEAR(y, y_exact(t), 1e-9);
  }
}
