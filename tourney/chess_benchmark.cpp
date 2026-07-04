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
    auto fstart = std::chrono::steady_clock::now();
    size_t fast_nodes = state.PerftFast(depth);
    auto fend = std::chrono::steady_clock::now();
    double fast_secs = std::chrono::duration<double>(fend - fstart).count();
    double fast_throughput = fast_nodes / fast_secs / 1e6;

    state = ChessState::initial_position();
    PerftSearch<ChessState> perft2(state);
    auto sstart = std::chrono::steady_clock::now();
    size_t slow_nodes = perft2.search(depth);
    auto send = std::chrono::steady_clock::now();
    double slow_secs = std::chrono::duration<double>(send - sstart).count();
    double slow_throughput = slow_nodes / slow_secs / 1e6;

    if (fast_nodes != slow_nodes) {
      std::println("ERROR: perft({}) nodes mismatch: FAST={} SLOW={}", depth,
                   fast_nodes, slow_nodes);
    }

    std::println("perft({})={}", depth, slow_nodes);
    std::println("  FAST: {:.0f} M nodes/s ({:.3f} s)", fast_throughput,
                 fast_secs);
    std::println("  SLOW: {:.0f} M nodes/s ({:.3f} s)", slow_throughput,
                 slow_secs);
    std::println("  SPEEDUP: {:.2f}x", slow_secs / fast_secs);
  }

  return 0;
}
