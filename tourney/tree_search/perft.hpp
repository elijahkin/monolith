#include <cstddef>

template <typename State>
class PerftSearch {
 private:
  using Move = State::MoveT;

  State& state_;

 public:
  explicit PerftSearch(State& state) : state_(state) {}

  // Recursively count nodes for debugging move generation
  size_t search(size_t depth) {
    if (depth == 0) {
      return 1;
    }

    size_t nodes = 0;
    Move buf[Move::kMaxMoves];
    size_t n = state_.FillLegalMoves(buf, Move::kMaxMoves);
    for (size_t i = 0; i < n; ++i) {
      state_.MakeMove(buf[i]);
      nodes += search(depth - 1);
      state_.UnmakeMove(buf[i]);
    }
    return nodes;
  }
};
