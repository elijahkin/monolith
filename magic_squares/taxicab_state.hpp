#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

// TODO It is possible this is doing redundant work with
// numerics/number_theory.hpp. Consider consolidating.

// Exponentiation using squaring-and-multiply
// template <typename T>
// T pow(T base, size_t exp) {
//   T res = 1;
//   while (exp > 0) {
//     if (exp % 2 == 1) {
//       res *= base;
//     }
//     base *= base;
//     exp /= 2;
//   }
//   return res;
// }

template <typename Power, typename Base>
class SumOfTwoPowersState {
 public:
  using PowersTable = std::shared_ptr<const std::vector<Power>>;

  static std::vector<SumOfTwoPowersState> seed(size_t k, Power limit) {
    auto powers = std::make_shared<std::vector<Power>>();
    for (Base a = 0;; ++a) {
      Power power = pow(static_cast<Power>(a), k);
      if (power > limit) {
        break;
      }
      powers->push_back(power);
    }
    // Initial state: 2 = 1^k + 1^k
    return {SumOfTwoPowersState(/*sum=*/2, /*a=*/1, /*b=*/1, std::move(powers),
                                limit)};
  }

  SumOfTwoPowersState(Power sum, Base a, Base b, PowersTable powers,
                      Power limit)
      : sum_(sum), a_(a), b_(b), powers_(std::move(powers)), limit_(limit) {}

  Power key() const { return sum_; }
  std::pair<Base, Base> witness() const { return {a_, b_}; }

  bool operator>(const SumOfTwoPowersState& other) const {
    return sum_ > other.sum_;
  }

  std::vector<SumOfTwoPowersState> spawn_successors() const {
    std::vector<SumOfTwoPowersState> out;
    const std::vector<Power>& powers = *powers_;

    // Path 1: Increment b (move right in the grid)
    Base next_b = b_ + 1;
    if (next_b < powers.size()) {
      Power next_sum_row = powers[a_] + powers[next_b];
      if (next_sum_row <= limit_) {
        out.emplace_back(next_sum_row, a_, next_b, powers_, limit_);
      }
    }

    // Path 2: Start a new row ONLY when we pop a diagonal element (a, a).
    // This ensures every pair (a, b) where a <= b is visited exactly once.
    if (a_ == b_) {
      Base next_a = a_ + 1;
      if (next_a < powers.size()) {
        Power next_sum_diag = powers[next_a] + powers[next_a];
        if (next_sum_diag <= limit_) {
          out.emplace_back(next_sum_diag, next_a, next_a, powers_, limit_);
        }
      }
    }

    return out;
  }

 private:
  Power sum_;
  Base a_, b_;
  PowersTable powers_;
  Power limit_;
};

template <typename Power, typename Base>
class DifferenceOfTwoPowersState {
 public:
  using PowersTable = std::shared_ptr<const std::vector<Power>>;

  static std::vector<DifferenceOfTwoPowersState> seed(size_t k, Power limit) {
    auto powers = std::make_shared<std::vector<Power>>();
    for (Base a = 0;; ++a) {
      Power power = pow(static_cast<Power>(a), k);
      if (power > limit) {
        break;
      }
      powers->push_back(power);
    }
    // Initial state: 1^k - 0^k = 1
    return {DifferenceOfTwoPowersState(/*diff=*/1, /*a=*/0, /*b=*/1,
                                       std::move(powers), limit)};
  }

  DifferenceOfTwoPowersState(Power diff, Base a, Base b, PowersTable powers,
                             Power limit)
      : diff_(diff), a_(a), b_(b), powers_(std::move(powers)), limit_(limit) {}

  Power key() const { return diff_; }
  // Witness is the larger base first: {b, a}.
  std::pair<Base, Base> witness() const { return {b_, a_}; }

  bool operator>(const DifferenceOfTwoPowersState& other) const {
    return diff_ > other.diff_;
  }

  std::vector<DifferenceOfTwoPowersState> spawn_successors() const {
    std::vector<DifferenceOfTwoPowersState> out;
    const std::vector<Power>& powers = *powers_;

    // Path 1: Increment b (move down the column).
    Base next_b = b_ + 1;
    if (next_b < powers.size()) {
      Power next_diff = powers[next_b] - powers[a_];
      if (next_diff <= limit_) {
        out.emplace_back(next_diff, a_, next_b, powers_, limit_);
      }
    }

    // Path 2: Start a new column ONLY for "leading edge" states (b == a + 1).
    // This guarantees every pair (a, b) with a < b is visited exactly once.
    if (b_ == a_ + 1) {
      Base next_a = a_ + 1;
      Base next_b = next_a + 1;
      if (next_b < powers.size()) {
        Power next_diff = powers[next_b] - powers[next_a];
        if (next_diff <= limit_) {
          out.emplace_back(next_diff, next_a, next_b, powers_, limit_);
        }
      }
    }

    return out;
  }

 private:
  Power diff_;
  Base a_, b_;
  PowersTable powers_;
  Power limit_;
};
