#include <concepts>
#include <type_traits>
#include <utility>

// ================================= Monoids ==================================
// A monoid is a set with an associative binary operation and an identity
// element. This is the minimal structure needed for fast exponentiation.

template <typename T, typename BinaryOp>
concept Monoid = std::regular<T> && std::invocable<BinaryOp, T, T> &&
                 std::convertible_to<std::invoke_result_t<BinaryOp, T, T>, T>;

// Computes op applied to `a` with itself `b` times, using binary
// exponentiation. `op` must be associative and `identity` a left-identity.
template <typename T, std::unsigned_integral U, typename BinaryOp>
  requires Monoid<T, BinaryOp>
T power(T a, U b, BinaryOp op, T identity) {
  T result = std::move(identity);
  while (b > 0) {
    if (b & 1) {
      result = op(result, a);
    }
    a = op(a, a);
    b >>= 1;
  }
  return result;
}

// ================================== Rings ===================================
// A ring is a set with addition, subtraction, multiplication, additive inverse,
// and distinguished elements 0 (additive identity) and 1 (multiplicative
// identity). Note: concepts enforce syntax only.

template <typename T>
concept Ring = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a* b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
  { T{0} } -> std::convertible_to<T>;
  { T{1} } -> std::convertible_to<T>;
  requires std::equality_comparable<T>;
};

// TODO Why do we have `Ring` AND `RingElement`?
template <typename T>
concept RingElement = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a* b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { T(0) };
};

// ============================ Euclidean Domains =============================
// A Euclidean domain is a ring equipped with a Euclidean function that enables
// division-with-remainder. Models must provide edivmod(a, b) returning {q, r}
// such that a = q*b + r, where r is "smaller" than b under that function.
//
// Known instances:
//   - std::integral types
//   - Univariate polynomials over a field (TODO)

template <std::integral T>
std::pair<T, T> edivmod(T a, T b) {
  T q = a / b;
  T r = a % b;
  if (r < T{0}) {
    // Adjust to floored division
    r += b;
    q -= T{1};
  }
  return {q, r};
}

template <typename T>
concept EuclideanDomain = Ring<T> && requires(T a, T b) {
  { edivmod(a, b) } -> std::convertible_to<std::pair<T, T>>;
};

template <EuclideanDomain T>
T ediv(T a, T b) {
  return edivmod(a, b).first;
}

template <EuclideanDomain T>
T emod(T a, T b) {
  return edivmod(a, b).second;
}

template <typename T>
concept OrderedField = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a* b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  { a < b } -> std::convertible_to<bool>;
  T{0};
};
