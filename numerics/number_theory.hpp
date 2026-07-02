#include <bitset>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "algebraic_concepts.hpp"

namespace nt {

// Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
template <EuclideanDomain T>
std::tuple<T, T, T> xgcd(T a, T b) {
  T old_r = a;
  T r = b;
  T old_s = T{1};
  T s = T{0};
  T old_t = T{0};
  T t = T{1};

  while (r != T{0}) {
    auto [q, rem] = edivmod(old_r, r);
    old_r = std::exchange(r, rem);
    old_s = std::exchange(s, old_s - (q * s));
    old_t = std::exchange(t, old_t - (q * t));
  }

  return {old_r, old_s, old_t};
}

// Given a list of congruences x ≡ a_i (mod n_i) where the n_i are pairwise
// coprime, returns the unique solution x modulo N = n_1 * ... * n_k.
template <EuclideanDomain T>
T crt(const std::vector<std::pair<T, T>>& eqs) {
  // A lambda function for combining two modular equations
  auto combine = [](std::pair<T, T> eq1, std::pair<T, T> eq2) {
    auto [a1, n1] = eq1;
    auto [a2, n2] = eq2;
    auto [g, p, q] = xgcd(n1, n2);
    T modulus = n1 * n2;
    return std::pair{emod((a1 * q * n2) + (a2 * p * n1), modulus), modulus};
  };

  auto [x, _] =
      std::accumulate(eqs.begin() + 1, eqs.end(), eqs.front(), combine);
  return x;
}

// Returns x such that a*x ≡ 1 (mod m), or nullopt if gcd(a,m) != 1.
template <EuclideanDomain T>
std::optional<T> modinv(T a, T m) {
  auto [g, x, _] = xgcd(a, m);
  if (g != T{1}) {
    return std::nullopt;
  }
  return emod(x, m);
}

// ============================== Miscellaneous ===============================

// TODO Instead of power mod, it would make more sense to add a `Zmod` ring, in
// which case this would simply become `power` as applied to the multiplication
// of that Ring?
template <std::unsigned_integral U>
U powmod(U base, U exponent, U modulus) {
  auto mod_mul = [modulus](U x, U y) { return mulmod(x, y, modulus); };
  return power(base % modulus, exponent, mod_mul, U{1} % modulus);
}

// TODO In the language of rings, what do these correspond to?
// Compute the k-th power residues mod n
template <std::unsigned_integral U>
std::set<U> kth_power_residues(U exponent, U modulus) {
  std::set<U> result;
  for (U a = 0; a < modulus; ++a) {
    result.insert(powmod(a, exponent, modulus));
  }
  return result;
}

// The sieve of Eratosthenes is intentionally integer-specific, since it relies
// on well-ordering to enumerate candidates up to sqrt(N).
template <size_t N>
std::bitset<N> sieve_of_eratosthenes() {
  std::bitset<N> is_prime;
  if (N < 2) {
    return is_prime;
  }

  // Assume initially that everything is prime
  is_prime.set();

  // Mark 0 and 1 as composite
  is_prime.reset(0);
  is_prime.reset(1);

  // Loop from 2 up to the square root of N
  for (size_t p = 2; p <= N / p; ++p) {
    // For each prime found, mark its multiples as composite
    if (is_prime.test(p)) {
      for (size_t i = p * p; i < N; i += p) {
        is_prime.reset(i);
      }
    }
  }
  return is_prime;
}

}  // namespace nt

// Deduce modular restrictions for the primitive solutions to \sum_{i = 1}^n
// a_i^k = \sum_{j = 1}^m b_j^k
//
// i.e. for k=6, n=2, m=6 at least three of the b_j are 0 mod 42
template <typename T>
void solve_lps(T exponent, T num_lhs, T num_rhs) {
  // Canonicalize: ensure num_lhs <= num_rhs for slightly cleaner output.
  if (num_lhs > num_rhs) {
    std::swap(num_lhs, num_rhs);
  }

  // TODO Probably only want prime power moduli here
  T max_modulus = 100;
  for (T m = T{2}; m < max_modulus; ++m) {
    auto res = nt::kth_power_residues(exponent, m);

    // auto lhs_sums = nt::power_sum_residues(res, num_lhs, m);
    // auto rhs_sums = nt::power_sum_residues(res, num_lhs, m);

    // The feasible sums are those achievable by both sides
    std::set<T> feasible;
    // TODO Intersection
  }

  // TODO Use CRT to stitch together across moduli
}
