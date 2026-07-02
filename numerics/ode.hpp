#include <array>
#include <concepts>
#include <cstddef>
#include <generator>
#include <utility>

namespace ode {

template <typename T, size_t s>
struct ButcherTableau {
  using value_type = T;
  static constexpr size_t kNumSteps = s;

  std::array<std::array<T, s>, s> a;
  std::array<T, s> b;
  std::array<T, s> c;
};

namespace methods {

template <typename T>
constexpr ode::ButcherTableau<T, 4> kRK4 = {
    .a = {{{0, 0, 0, 0},
           {T(1) / T(2), 0, 0, 0},
           {0, T(1) / T(2), 0, 0},
           {0, 0, 1, 0}}},
    .b = {T(1) / T(6), T(1) / T(3), T(1) / T(3), T(1) / T(6)},
    .c = {0, T(1) / T(2), T(1) / T(2), 1}};

}  // namespace methods

// TODO Would it make more sense to pass in the final t value?
// TODO Can we use concepts to require that `State` supports addition, etc?
template <typename Time, typename State>
struct IntegrationParams {
  Time t0;
  State y0;
  Time h;
  size_t num_iters;
};

// Explicit Runge-Kutta solver for ODE dy/dt = f(t, y) with y(t0) = y0
// TODO It would be nice if we eventually support implicit methods
// TODO Can we pass method as a parameter? Would it hurt performance?
template <ButcherTableau method, typename Time, typename State>
std::generator<std::pair<Time, State>> solve_ivp(
    std::invocable<Time, State> auto f, IntegrationParams<Time, State> p) {
  Time t = p.t0;
  State y = p.y0;
  co_yield {t, y};

  for (size_t n = 0; n < p.num_iters; ++n) {
    std::array<State, method.kNumSteps> k{};
    for (size_t i = 0; i < method.kNumSteps; ++i) {
      State sum_j = 0;
      for (size_t j = 0; j < i; ++j) {
        sum_j += method.a[i][j] * k[j];
      }
      k[i] = f(t + (method.c[i] * p.h), y + (p.h * sum_j));
    }

    State sum_i = 0;
    for (size_t i = 0; i < method.kNumSteps; ++i) {
      sum_i += method.b[i] * k[i];
    }

    y += p.h * sum_i;
    t += p.h;
    co_yield {t, y};
  }
}

}  // namespace ode
