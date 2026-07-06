#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

const std::string kCursorHome = "\x1B[H";
const std::string kEraseScreen = "\x1B[2J";

enum class TtPiece : uint8_t { kEmpty, kX, kO };

struct TicTacToeMove {
  static constexpr int kMaxMoves = 9;

  uint8_t square;
};

// TODO Need to add a test for this class
class TicTacToeState final {
 public:
  using MoveT = TicTacToeMove;

  TicTacToeState() = default;

  void MakeMove(const TicTacToeMove& move) {
    board_[move.square] = (x_to_move_ ? TtPiece::kX : TtPiece::kO);
    x_to_move_ = !x_to_move_;
  }

  void UnmakeMove(const TicTacToeMove& move) {
    board_[move.square] = TtPiece::kEmpty;
    x_to_move_ = !x_to_move_;
  }

  [[nodiscard]] size_t FillLegalMoves(TicTacToeMove* buffer,
                                      size_t capacity) const {
    size_t n = 0;
    for (uint8_t i = 0; i < 9 && n < capacity; ++i) {
      if (board_[i] == TtPiece::kEmpty) {
        buffer[n++] = TicTacToeMove{.square = i};
      }
    }
    return n;
  }

  [[nodiscard]] bool IsOver() const {
    auto win = [&](int a, int b, int c) {
      return board_[a] != TtPiece::kEmpty && board_[a] == board_[b] &&
             board_[b] == board_[c];
    };
    for (int i = 0; i < 3; ++i) {
      if (win(i, i + 3, i + 6) || win(3 * i, (3 * i) + 1, (3 * i) + 2)) {
        return true;
      }
    }
    if (win(0, 4, 8) || win(2, 4, 6)) {
      return true;
    }
    for (auto p : board_) {
      if (p == TtPiece::kEmpty) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] std::optional<TicTacToeMove> Parse(
      const std::string& input) const;

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
