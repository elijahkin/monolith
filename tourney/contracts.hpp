#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

// Defines the necessary functions to implement a game.
template <typename Move>
class Game {
 public:
  virtual ~Game() = default;

  virtual void MakeMove(const Move &move) = 0;

  virtual void UnmakeMove(const Move &move) = 0;

  [[nodiscard]] virtual int FillLegalMoves(Move *buffer,
                                           int capacity) const = 0;

  [[nodiscard]] virtual bool IsOver() const = 0;

  [[nodiscard]] virtual std::string ToString() const = 0;

  [[nodiscard]] virtual std::optional<Move> Parse(
      const std::string &input) const = 0;

  // Recursively count nodes for debugging move generation
  [[nodiscard]] size_t Perft(size_t depth) {
    if (depth == 0) {
      return 1;
    }

    size_t nodes = 0;
    Move buf[Move::kMaxMoves];
    int n = FillLegalMoves(buf, Move::kMaxMoves);
    for (int i = 0; i < n; ++i) {
      MakeMove(buf[i]);
      nodes += Perft(depth - 1);
      UnmakeMove(buf[i]);
    }
    return nodes;
  }
};

// Defines the necessary functions to implement an agent.
template <typename Move>
class Agent {
 public:
  explicit Agent(Game<Move> &state) : state_(state) {}

  virtual ~Agent() = default;

  [[nodiscard]] virtual Move SelectMove() = 0;

 protected:
  Game<Move> &state_;
};
