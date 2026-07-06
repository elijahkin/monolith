// Usage: bazel run //tourney:chess_benchmark

#include <chrono>
#include <cstddef>
#include <print>

#include "game_states/chess_state.hpp"
#include "tree_search/perft.hpp"

int main() {
  auto state = ChessState::initial_position();
  PerftSearch<ChessState> perft(state);

  for (size_t depth = 1; depth <= 6; ++depth) {
    state = ChessState::initial_position();
    auto start = std::chrono::steady_clock::now();
    size_t nodes = perft.search(depth);
    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    double throughput = nodes / secs / 1e6;

    std::println("perft({})={}", depth, nodes);
    std::println("  {:.0f} M nodes/s ({:.3f} s)", throughput, secs);
    std::println("");
  }

  return 0;
}
