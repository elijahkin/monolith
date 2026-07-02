#include <algorithm>
#include <cstddef>
#include <limits>

template <typename State>
class Minimax {
 public:
  using Value = decltype(std::declval<const State>().evaluate());

  Value search(State state, size_t depth) {
    return negamax(state, depth, std::numeric_limits<Value>::lowest(),
                   std::numeric_limits<Value>::max());
  }

 private:
  Value negamax(State state, size_t depth, Value alpha, Value beta) {
    if (depth == 0 || state.is_terminal()) {
      return state.evaluate();
    }

    Value best = std::numeric_limits<Value>::lowest();
    for (const auto& move : state.legal_moves()) {
      auto undo = state.make_move(move);
      Value val = -negamax(state, depth - 1, -beta, -alpha);
      state.unmake_move(move, undo);
      best = std::max(best, val);
      if (best >= beta) {
        return best;
      }
      alpha = std::max(alpha, best);
    }
    return best;
  }
};
