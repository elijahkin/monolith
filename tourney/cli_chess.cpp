// Usage: bazel run //tourney:cli_chess

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "agents/human_agent.hpp"
#include "agents/minimax_agent.hpp"
#include "contracts.hpp"
#include "game_states/chess_state.hpp"

std::function<Score(const ChessMove&)> kBlackAdvantageOnCapture =
    [](const ChessMove& move) {
      constexpr std::array<Score, 7> kMaterialValues = {0, 200, 9, 5, 3, 3, 1};
      return kMaterialValues[static_cast<size_t>(move.captured)];
    };

int main() {
  auto game = ChessState::initial_position();

  std::vector<std::unique_ptr<Agent<ChessMove>>> agents;
  agents.push_back(std::make_unique<HumanAgent<ChessMove>>(game));
  agents.push_back(std::make_unique<MinimaxAgent<ChessMove>>(
      game, 5, kBlackAdvantageOnCapture));

  while (true) {
    for (const auto& agent_ptr : agents) {
      std::cout << game.ToString() << "\n";
      if (game.IsOver()) {
        return 0;
      }
      const auto move = agent_ptr->SelectMove();
      game.RecordMove(move);
      game.MakeMove(move);
    }
  }
}
