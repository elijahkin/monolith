#include <cstddef>

template <typename State>
class Perft {
 public:
  size_t search(State& state, size_t depth) {
    if (depth == 0) {
      return 1;
    }
    size_t nodes = 0;
    for (const auto& m : state.legal_moves()) {
      auto undo = state.make_move(m);
      nodes += search(state, depth - 1);
      state.unmake_move(m, undo);
    }
    return nodes;
  }
};
