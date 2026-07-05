#pragma once

#include <cstddef>
#include <optional>
#include <string>

// TODO It might be better if the contracts are concepts rather than classes.
// Would this eliminate virtual dispatching cost?

// Defines the necessary functions to implement a game.
template <typename Move>
class Game {
 public:
  virtual ~Game() = default;

  virtual void MakeMove(const Move& move) = 0;

  virtual void UnmakeMove(const Move& move) = 0;

  [[nodiscard]] virtual size_t FillLegalMoves(Move* buffer,
                                              size_t capacity) const = 0;

  [[nodiscard]] virtual bool IsOver() const = 0;

  [[nodiscard]] virtual std::string ToString() const = 0;

  [[nodiscard]] virtual std::optional<Move> Parse(
      const std::string& input) const = 0;
};

// Defines the necessary functions to implement an agent.
template <typename Move>
class Agent {
 public:
  explicit Agent(Game<Move>& state) : state_(state) {}

  virtual ~Agent() = default;

  [[nodiscard]] virtual Move SelectMove() = 0;

 protected:
  Game<Move>& state_;
};
