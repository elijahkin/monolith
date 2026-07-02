
#include <algorithm>
#include <array>
#include <cstddef>
#include <generator>
#include <ostream>

#include "algebraic_concepts.hpp"

namespace rootfinding {

// ============================ Bracketing Methods ============================

template <OrderedField T>
struct Bracket {
  T a, b;
  T fa, fb;

  friend std::ostream& operator<<(std::ostream& os, const Bracket& bracket) {
    return os << '(' << bracket.a << ", " << bracket.b << ')';
  }

  T midpoint() const { return a + ((b - a) / 2); }

  T secant_point() const { return (a * fb - b * fa) / (fb - fa); }

  void update(T c, T fc) {
    if ((fa < T{0}) != (fc < T{0})) {
      b = c;
      fb = fc;
    } else {
      a = c;
      fa = fc;
    }
  }
};

template <typename FuncT, OrderedField T, typename SelectorT>
std::generator<Bracket<T>> bracket_method(FuncT f, T a, T b,
                                          SelectorT next_point) {
  Bracket<T> bracket = {a, b, f(a), f(b)};
  while (true) {
    co_yield bracket;
    const T c = next_point(bracket);
    if (c == bracket.a || c == bracket.b) {
      co_return;
    }
    bracket.update(c, f(c));
  }
}

// Bisection method: Linear, order 1. log2((b-a)/tol) steps to reach a given
// tolerance. Completely robust, it cannot diverge or stagnate as long as the
// initial bracket [a, b] contains a root.
template <typename FuncT, OrderedField T>
auto bisection(FuncT f, T a, T b) {
  return bracket_method(
      f, a, b, [](const Bracket<T>& bracket) { return bracket.midpoint(); });
}

// False position method: Linear, order 1. Typically faster than bisection in
// practice. However, if f is convex on [a, b], one endpoint becomes "sticky"
// and is never updated, so the bracket shrinks from one side only.
template <typename FuncT, OrderedField T>
auto false_position(FuncT f, T a, T b) {
  return bracket_method(f, a, b, [](const Bracket<T>& bracket) {
    return bracket.secant_point();
  });
}

// TODO Illinois, Pegasus, Brent?

// ============================ Iterative Methods =============================

template <OrderedField T, size_t N>
struct Window {
  std::array<T, N> x;
  std::array<T, N> fx;

  void update(T x_new, T fx_new) {
    std::shift_left(x.begin(), x.end(), 1);
    std::shift_left(fx.begin(), fx.end(), 1);
    x.back() = x_new;
    fx.back() = fx_new;
  }
};

template <OrderedField T, std::size_t N, typename FuncT, typename StepT>
std::generator<T> iterative_method(std::array<T, N> x0, FuncT f, StepT step,
                                   T eps = T{1e-12}) {
  Window<T, N> window;
  for (size_t i = 0; i < N; ++i) {
    window.x[i] = x0[i];
    window.fx[i] = f(x0[i]);
  }
  while (true) {
    const T x_new = step(window);
    co_yield x_new;
    T diff = x_new - window.x.back();
    if (diff < T{0}) {
      diff = T{0} - diff;
    }
    if (diff < eps) {
      co_return;
    }
    window.update(x_new, f(x_new));
  }
}

// Newton's method: Quadratic, order 2. Requires an analytic derivative, can
// diverge if x0 is far from the root or if f has inflection points nearby, and
// stalls with only linear convergence at repeated roots. One evaluation of f
// and df per iteration.
template <typename FuncT, typename DerivT, OrderedField T>
auto newton(FuncT f, T x0, DerivT df) {
  return iterative_method(std::array{x0}, f, [df](const Window<T, 1>& w) {
    return w.x[0] - (w.fx[0] / df(w.x[0]));
  });
}

// Secant method: Superlinear, order ~1.618. Almost as fast as Newton in
// practice, without needing a derivative. Like Newton, it can diverge from poor
// starting points and slowdown at repeated roots. One evaluation of f per
// iteration.
template <typename FuncT, OrderedField T>
auto secant(FuncT f, T x0, T x1) {
  return iterative_method(std::array{x0, x1}, f, [](const Window<T, 2>& w) {
    return w.x[1] - (w.fx[1] * (w.x[1] - w.x[0]) / (w.fx[1] - w.fx[0]));
  });
}

// Steffensen's method: Quadratic, order 2. Does not require a derivative. Two
// evaluations of f per iteration.
template <typename FuncT, OrderedField T>
auto steffensen(FuncT f, T x0) {
  return iterative_method(std::array{x0}, f, [f](const Window<T, 1>& w) {
    const T fx = f(w.x[0]);
    return (fx * fx) / (f(w.x[0] + fx) - fx);
  });
}

}  // namespace rootfinding
