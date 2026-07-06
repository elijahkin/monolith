#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "../contracts.hpp"

using Score = double;

template <GameState State>
class MinimaxSearch {
 private:
  using Move = typename State::MoveT;

  State& state_;
  int max_plies_;
  Score heuristic_value_ = 0;
  std::function<Score(const Move&)> heuristic_value_adjustment_;
  size_t leaf_nodes_count_ = 0;
  Score best_value_ = 0;

  static constexpr Score kInf = std::numeric_limits<Score>::infinity();
  static constexpr Score kNegInf = -std::numeric_limits<Score>::infinity();

  // https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  Score AlphaBeta(const Move& move, int ply, Score alpha, Score beta) {
    typename State::MoveUndo undo;
    state_.MakeMove(move, undo);
    heuristic_value_ += heuristic_value_adjustment_(move);

    if (ply == max_plies_) {
      const Score leaf_value = heuristic_value_;
      leaf_nodes_count_++;
      state_.UnmakeMove(move, undo);
      heuristic_value_ -= heuristic_value_adjustment_(move);
      return leaf_value;
    }

    Score value;
    if (ply % 2 == 0) {  // Maximizing player
      value = kNegInf;
      Move buf[Move::kMaxMoves];
      size_t n = state_.FillLegalMoves(buf, Move::kMaxMoves);
      for (size_t i = 0; i < n; ++i) {
        value = std::max(value, AlphaBeta(buf[i], ply + 1, alpha, beta));
        if (value >= beta) {
          break;
        }
        alpha = std::max(alpha, value);
      }
    } else {  // Minimizing player
      value = kInf;
      Move buf[Move::kMaxMoves];
      size_t n = state_.FillLegalMoves(buf, Move::kMaxMoves);
      for (size_t i = 0; i < n; ++i) {
        value = std::min(value, AlphaBeta(buf[i], ply + 1, alpha, beta));
        if (value <= alpha) {
          break;
        }
        beta = std::min(beta, value);
      }
    }
    state_.UnmakeMove(move, undo);
    heuristic_value_ -= heuristic_value_adjustment_(move);
    return value;
  }

 public:
  MinimaxSearch(State& state, int max_plies,
                std::function<Score(const Move&)> heuristic_value_adjustment)
      : state_(state),
        max_plies_(max_plies),
        heuristic_value_adjustment_(heuristic_value_adjustment) {}

  Move Search() {
    leaf_nodes_count_ = 0;
    best_value_ = kNegInf;
    std::vector<Move> best_moves;
    Move buf[Move::kMaxMoves];
    size_t n = state_.FillLegalMoves(buf, Move::kMaxMoves);
    for (size_t i = 0; i < n; ++i) {
      const Score value = AlphaBeta(buf[i], 1, kNegInf, kInf);
      if (value > best_value_) {
        best_moves.clear();
        best_moves.push_back(buf[i]);
        best_value_ = value;
      } else if (value == best_value_) {
        best_moves.push_back(buf[i]);
      }
    }
    return best_moves[0];
  }

  Score BestValue() const { return best_value_; }
  [[nodiscard]] size_t LeafNodesCount() const { return leaf_nodes_count_; }
};
