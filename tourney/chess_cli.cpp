// Usage: bazel run //tourney:chess_cli

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "games/chess.hpp"
#include "players.hpp"

std::function<Score(const ChessMove&)> black_advantage_on_capture =
    [](const ChessMove& move) {
      constexpr std::array<Score, 7> kMaterialValues = {0, 200, 9, 5, 3, 3, 1};
      return kMaterialValues[static_cast<size_t>(move.captured)];
    };

int main() {
  auto game = ChessState::initial_position();

  std::vector<std::unique_ptr<PlayerBase<ChessMove>>> agents;
  agents.push_back(std::make_unique<HumanPlayer<ChessState>>(game));
  agents.push_back(std::make_unique<MinimaxPlayer<ChessState>>(
      game, 5, black_advantage_on_capture));

  while (true) {
    for (const auto& agent_ptr : agents) {
      std::cout << game.ToString() << "\n";
      if (game.IsOver()) {
        return 0;
      }
      const auto move = agent_ptr->SelectMove();
      game.RecordMove(move);
      ChessState::MoveUndo undo;
      game.MakeMove(move, undo);
    }
  }
}
