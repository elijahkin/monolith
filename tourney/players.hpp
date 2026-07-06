#include <chrono>
#include <functional>
#include <iostream>
#include <string>

#include "contracts.hpp"
#include "tree_search/minimax.hpp"

template <typename Move>
class Player {
 public:
  explicit Player(Game<Move>& state) : state_(state) {}

  virtual ~Player() = default;

  [[nodiscard]] virtual Move SelectMove() = 0;

 protected:
  Game<Move>& state_;
};

template <typename Move>
class HumanPlayer final : public Player<Move> {
 public:
  explicit HumanPlayer(Game<Move>& state) : Player<Move>(state) {}

  Move SelectMove() override {
    std::string input;
    while (true) {
      std::cout << "Please enter a move: ";
      std::cin >> input;
      auto parsed = this->state_.Parse(input);
      if (parsed.has_value()) {
        return parsed.value();
        break;
      }
      std::cout << "Invalid entry! ";
    }
  }
};

template <typename State>
class MinimaxPlayer final : public Player<typename State::MoveT> {
  using Move = typename State::MoveT;

 public:
  MinimaxPlayer(State& state, int max_plies,
                std::function<Score(const Move&)> heuristic_value_adjustment)
      : Player<Move>(state),
        search_(state, max_plies, heuristic_value_adjustment) {}

  Move SelectMove() override {
    std::cout << "Minimax player is thinking...\n";
    const auto begin = std::chrono::steady_clock::now();
    Move move = search_.Search();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();
    std::cout << "Selected move with value " << search_.BestValue()
              << " after visiting " << search_.LeafNodesCount()
              << " leaf nodes in " << elapsed_ms << "ms\n";
    return move;
  }

 private:
  MinimaxSearch<State> search_;
};
