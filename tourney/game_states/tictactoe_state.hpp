#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../contracts.hpp"

const std::string kCursorHome = "\x1B[H";
const std::string kEraseScreen = "\x1B[2J";

enum class TtPiece : uint8_t { kEmpty, kX, kO };

struct TicTacToeMove {
  uint8_t square;
};

// TODO Need to add a test for this class
class TicTacToeState final : public Game<TicTacToeMove> {
 public:
  TicTacToeState() = default;

  void MakeMove(const TicTacToeMove& move) override {
    board_[move.square] = (x_to_move_ ? TtPiece::kX : TtPiece::kO);
    x_to_move_ = !x_to_move_;
  }

  void UnmakeMove(const TicTacToeMove& move) override {
    board_[move.square] = TtPiece::kEmpty;
    x_to_move_ = !x_to_move_;
  }

  [[nodiscard]] std::vector<TicTacToeMove> GenerateLegalMoves() const override {
    std::vector<TicTacToeMove> moves;
    for (uint8_t i = 0; i < 9; ++i) {
      if (board_[i] == TtPiece::kEmpty) {
        moves.push_back(TicTacToeMove{.square = i});
      }
    }
    return moves;
  }

  [[nodiscard]] std::string ToString() const override;

  [[nodiscard]] std::optional<TicTacToeMove> Parse(
      const std::string& input) const override;

 private:
  std::array<TtPiece, 9> board_{};
  bool x_to_move_ = true;
};

std::string TicTacToeState::ToString() const {
  static constexpr std::array<char, 3> kPieceLetters = {' ', 'X', 'O'};
  std::string output = kEraseScreen + kCursorHome;
  auto p = [&](int i) { return kPieceLetters[static_cast<size_t>(board_[i])]; };
  output += p(0);
  output += " \u2502";
  output += p(1);
  output += " \u2502";
  output += p(2);
  output += " \n\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\n";
  output += p(3);
  output += " \u2502";
  output += p(4);
  output += " \u2502";
  output += p(5);
  output += " \n\u2500\u2500\u253c\u2500\u2500\u253c\u2500\u2500\n";
  output += p(6);
  output += " \u2502";
  output += p(7);
  output += " \u2502";
  output += p(8);
  output += " \n";
  return output;
}

std::optional<TicTacToeMove> TicTacToeState::Parse(
    const std::string& input) const {
  const auto i = static_cast<uint8_t>(std::stoi(input));
  if (i >= 9 || board_[i] != TtPiece::kEmpty) {
    return std::nullopt;
  }
  return TicTacToeMove{i};
}
