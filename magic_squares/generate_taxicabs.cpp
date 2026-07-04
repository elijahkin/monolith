// Usage: bazel run //magic_squares:generate_taxicabs 3 2 2 8

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "../util/io.hpp"
#include "collision_generator.hpp"
#include "taxicab_state.hpp"

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

  // TODO I think we are ignoring `n` currently, and acting as if it's always 2.
  using State = SumOfTwoPowersState<__uint128_t, size_t>;
  CollisionGenerator<State> generator(j, State::seed(k, limit));

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
