#include <cstddef>

// Perft (performance test, move path enumeration) simply walks over a game tree
// to count all the leaf nodes of a certain depth. It is primarily used for
// ensuring the correctness and measuring the performance of move generation.
//
// https://www.chessprogramming.org/Perft

template <typename State>
class PerftSearch {
 private:
  using Move = State::MoveT;

  State& state_;

 public:
  explicit PerftSearch(State& state) : state_(state) {}

  size_t search(size_t depth) {
    // Depth 0 corresponds to only the root node.
    if (depth == 0) {
      return 1;
    }

    Move moves[Move::kMaxMoves];
    size_t n = state_.FillLegalMoves(moves, Move::kMaxMoves);

    // For depth 1, just return the number of legal moves. Note that this
    // condition is technically unnecessary, but eliminates the cost of
    // MakeMove, UnmakeMove, and recursion.
    if (depth == 1) {
      return n;
    }

    // Otherwise, we iterate over the moves, making each one, recursing, and
    // unmaking it before moving on to the next.
    size_t nodes = 0;
    for (size_t i = 0; i < n; ++i) {
      state_.MakeMove(moves[i]);
      nodes += search(depth - 1);
      state_.UnmakeMove(moves[i]);
    }
    return nodes;
  }
};
