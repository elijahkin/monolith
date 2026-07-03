// Usage: bazel run //magic_squares:taxicab 3 2 2 8

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "../util/io.hpp"

// TODO It is possible this is doing redundant work with
// numerics/number_theory.hpp. Consider consolidating.

// Exponentiation using squaring-and-multiply
template <typename T>
T pow(T base, size_t exp) {
  T res = 1;
  while (exp > 0) {
    if (exp % 2 == 1) {
      res *= base;
    }
    base *= base;
    exp /= 2;
  }
  return res;
}

template <typename Power, typename Base>
class TaxicabGenerator {
 public:
  struct Result {
    // TODO Is it worth including sum in the result? We could also consider
    // using some kind of structured output (e.g. JSON) rather than plain text.
    Power sum;
    std::vector<std::pair<Base, Base>> pairs;

    [[nodiscard]] std::string to_string() const {
      std::string s;
      bool first = true;
      for (const auto& [a, b] : pairs) {
        if (!first) {
          s += ' ';
        }
        s += std::to_string(a) + ' ' + std::to_string(b);
        first = false;
      }
      return s;
    }
  };

  TaxicabGenerator(size_t k, size_t num_ways, Power limit)
      : k_(k), num_ways_(num_ways), limit_(limit) {
    // Precompute powers: `limit` will never be large enough to cause a problem
    for (Base a = 0;; ++a) {
      Power power = pow(static_cast<Power>(a), k_);
      if (power > limit) {
        break;
      }
      powers_.push_back(power);
    }

    // Initial state: 2 = 1^k + 1^k
    pq_.push({2, 1, 1});
  }

  // Lazily-evaluates the next taxicab number
  std::optional<Result> next() {
    std::vector<std::pair<Base, Base>> current_pairs;
    current_pairs.reserve(10);  // Avoid repeated allocations

    while (!pq_.empty()) {
      Power current_sum = pq_.top().sum;
      current_pairs.clear();

      while (!pq_.empty() && pq_.top().sum == current_sum) {
        State top = pq_.top();
        pq_.pop();
        current_pairs.push_back({top.a, top.b});

        // Path 1: Increment b (move right in the grid)
        Base next_b = top.b + 1;
        if (next_b < powers_.size()) {
          Power next_sum_row = powers_[top.a] + powers_[next_b];
          if (next_sum_row <= limit_) {
            pq_.push({next_sum_row, top.a, next_b});
          }
        }

        // Path 2: Start a new row ONLY when we pop a diagonal element (a, a)
        // This ensures every pair (a, b) where a <= b is visited exactly once
        if (top.a == top.b) {
          Base next_a = top.a + 1;
          if (next_a < powers_.size()) {
            Power next_sum_diag = powers_[next_a] + powers_[next_a];
            if (next_sum_diag <= limit_) {
              pq_.push({next_sum_diag, next_a, next_a});
            }
          }
        }
      }

      // Report if we found a taxicab number
      if (current_pairs.size() >= num_ways_) {
        return Result{current_sum, current_pairs};
      }
    }
    // Queue is empty, no more results
    return std::nullopt;
  }

 private:
  // Represents a term in the priority queue: sum = a^3 + b^3
  struct State {
    Power sum;
    Base a;
    Base b;

    // Min-heap comparator (smallest sum on top)
    bool operator>(const State& other) const { return sum > other.sum; }
  };

  size_t k_, num_ways_;
  Power limit_;
  std::vector<Power> powers_;
  std::priority_queue<State, std::vector<State>, std::greater<>> pq_;
};

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <k> <j> <n> <digits>\n"
              << "  k       power to raise bases to (e.g. 3 for cubes)\n"
              << "  j       number of pairs per taxicab number (must be 2)\n"
              << "  n       number of solutions to find\n"
              << "  digits  upper bound is 10^digits\n";
    exit(1);
  }

  // Parse the command-line arguments
  const size_t k = std::stoull(argv[1]);
  const size_t j = std::stoull(argv[2]);
  const size_t n = std::stoull(argv[3]);
  const size_t digits = std::stoull(argv[4]);

  // TODO Eventually we should remove this restriction
  if (j != 2) {
    std::cout << "Only j=2 is supported currently, please retry.\n";
    exit(1);
  }

  if (n < 2) {
    // n == 1 will flood the output with trivial solutions
    std::cout << "Please specify n >= 2, please retry.\n";
    exit(1);
  }

  constexpr __uint128_t kTen = 10;
  const auto limit = pow(kTen, digits);

  TaxicabGenerator<__uint128_t, size_t> generator(k, n, limit);
  size_t num_sols = 0;
  std::cout << "Done with precomputation.\n";

  // Prepare the file where output will be written
  const std::string filename = "taxicab_" + std::to_string(k) + "_" +
                               std::to_string(j) + "_" + std::to_string(n) +
                               ".txt";

  std::ofstream outfile = io::GetOutputFileStream(
      io::ResolveOutputPath("magic_squares/log"), filename);

  // Perform the search, recording start and end timestamps
  const auto start = std::chrono::steady_clock::now();

  outfile << "taxicab " << k << ' ' << j << ' ' << n << ' ' << digits << '\n';
  while (auto result = generator.next()) {
    outfile << result->to_string() << '\n';
    num_sols++;
  }

  // Close the file and report the results of the search
  const auto end = std::chrono::steady_clock::now();
  outfile.close();
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Found " << num_sols << " solutions in " << elapsed_ms
            << "ms!\n";
  return 0;
}
