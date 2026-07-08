#include <chrono>
#include <functional>
#include <iostream>
#include <string>

#include "algorithms/minimax.hpp"
#include "contracts.hpp"

// TODO Exists only so that `chess_cli` can hold a heterogeneous collection of
// players (HumanPlayer + MinimaxPlayer) in one
// `std::vector<unique_ptr<PlayerBase<Move>>>`. Is this the right design choice?
// Maybe a concept for players could make sense as well?
template <typename Move>
class PlayerBase {
 public:
  explicit PlayerBase() = default;
  virtual ~PlayerBase() = default;

  [[nodiscard]] virtual Move SelectMove() = 0;
};

template <GameState State>
class StdinPlayer final : public PlayerBase<typename State::MoveT> {
  using Move = typename State::MoveT;

 public:
  explicit StdinPlayer(State& state) : state_(state) {}

  Move SelectMove() override {
    std::string input;
    while (true) {
      std::cout << "Please enter a move: ";
      std::cin >> input;
      auto parsed = state_.Parse(input);
      if (parsed.has_value()) {
        return parsed.value();
      }
      std::cout << "Invalid entry! ";
    }
  }

 private:
  State& state_;
};

template <GameState State>
class MinimaxPlayer final : public PlayerBase<typename State::MoveT> {
  using Move = typename State::MoveT;

 public:
  MinimaxPlayer(State& state, int max_plies,
                std::function<Score(const Move&)> heuristic_value_adjustment)
      : search_(state, max_plies, heuristic_value_adjustment) {}

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
