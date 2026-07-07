// Usage: bazel run //tourney:chess_cli

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "games/chess.hpp"
#include "players.hpp"

namespace {

using Piece = PieceType;

struct Glyph {
  std::array<std::array<const char*, 7>, 2> table = []() {
    std::array<std::array<const char*, 7>, 2> t{};
    t[0] = {
        /* kNone   */ " ",
        /* kPawn   */ "\u2659",
        /* kKnight */ "\u2658",
        /* kBishop */ "\u2657",
        /* kRook   */ "\u2656",
        /* kQueen  */ "\u2655",
        /* kKing   */ "\u2654",
    };
    t[1] = {
        /* kNone   */ " ",
        /* kPawn   */ "\u265F",
        /* kKnight */ "\u265E",
        /* kBishop */ "\u265D",
        /* kRook   */ "\u265C",
        /* kQueen  */ "\u265B",
        /* kKing   */ "\u265A",
    };
    return t;
  }();
};

constexpr Glyph kGlyphs;

constexpr std::array<char, 7> kPieceLetters = {'\0', '\0', 'N', 'B',
                                               'R',  'Q',  'K'};

class BoardFromFen {
 public:
  explicit BoardFromFen(std::string_view placement) {
    int rank = 7;
    int file = 0;
    for (const char c : placement) {
      if (c == '/') {
        --rank;
        file = 0;
        continue;
      }
      if (c >= '1' && c <= '8') {
        file += c - '0';
        continue;
      }
      Piece pt = PieceType::kNone;
      switch (c) {
        case 'P':
        case 'p':
          pt = PieceType::kPawn;
          break;
        case 'N':
        case 'n':
          pt = PieceType::kKnight;
          break;
        case 'B':
        case 'b':
          pt = PieceType::kBishop;
          break;
        case 'R':
        case 'r':
          pt = PieceType::kRook;
          break;
        case 'Q':
        case 'q':
          pt = PieceType::kQueen;
          break;
        case 'K':
        case 'k':
          pt = PieceType::kKing;
          break;
        default:
          break;
      }
      const int sq = (rank * 8) + file;
      cells_[sq] = {pt, c >= 'A' && c <= 'Z'};
      ++file;
    }
  }

  [[nodiscard]] std::pair<Piece, bool> at(int sq) const { return cells_[sq]; }

 private:
  std::array<std::pair<Piece, bool>, 64> cells_{};
};

struct FenParts {
  std::string_view placement;
  std::string_view side_to_move;
};
FenParts split_fen(std::string_view fen) {
  const auto space1 = fen.find(' ');
  if (space1 == std::string_view::npos) {
    return {.placement = fen, .side_to_move = "w"};
  }
  const auto side_start = space1 + 1;
  const auto space2 = fen.find(' ', side_start);
  std::string_view side = (space2 == std::string_view::npos)
                              ? fen.substr(side_start)
                              : fen.substr(side_start, space2 - side_start);
  return {.placement = fen.substr(0, space1), .side_to_move = side};
}

[[nodiscard]] std::string to_san(const ChessMove& move) {
  std::string out;
  out += kPieceLetters[static_cast<size_t>(move.piece)];
  if (move.captured != PieceType::kNone) {
    out += 'x';
  }
  out += static_cast<char>('a' + (move.to % 8));
  out += std::to_string(1 + (move.to / 8));
  return out;
}

class MoveHistory {
 public:
  void Record(int fullmove_number, const std::string& san) {
    if (fullmove_number > static_cast<int>(rows_.size())) {
      rows_.emplace_back().first = san;
    } else {
      rows_.back().second = san;
    }
  }

  [[nodiscard]] std::string row_str(std::size_t zero_based) const {
    if (zero_based >= rows_.size()) {
      return "";
    }
    const auto& row = rows_[zero_based];
    std::string move_str = std::to_string(zero_based + 1);
    move_str.insert(0, 3 - move_str.size(), ' ');
    move_str += ". ";
    move_str += row.first;
    if (!row.second.empty()) {
      move_str += ' ';
      move_str += row.second;
    }
    move_str.insert(move_str.end(), 14 - move_str.size(), ' ');
    return move_str;
  }

  [[nodiscard]] std::size_t row_count() const { return rows_.size(); }

 private:
  struct Row {
    std::string first;
    std::string second;
  };
  std::vector<Row> rows_;
};

[[nodiscard]] std::string RenderGame(const ChessState& state,
                                     const MoveHistory& history) {
  constexpr std::string_view kCursorHome = "\x1B[H";
  constexpr std::string_view kEraseScreen = "\x1B[2J";
  constexpr std::string_view kForegroundBlack = "\x1B[30m";
  constexpr std::string_view kForegroundGray = "\x1B[38;5;240m";
  constexpr std::string_view kForegroundDefault = "\x1B[39m";
  constexpr std::string_view kBackgroundMagenta = "\x1B[45m";
  constexpr std::string_view kBackgroundWhite = "\x1B[47m";
  constexpr std::string_view kBackgroundDefault = "\x1B[49m";

  const std::string fen = state.ToString();
  const auto fen_parts = split_fen(fen);
  const BoardFromFen board(fen_parts.placement);

  std::string output = std::string(kEraseScreen) + std::string(kCursorHome);
  for (int row = 0; row < 9; ++row) {
    const char rank = static_cast<char>('8' - row);
    output += (row == 8 ? ' ' : rank);
    output += ' ';
    for (int col = 0; col < 8; ++col) {
      const char file = static_cast<char>('a' + col);
      if (row != 8) {
        const int sq = (8 * (7 - row)) + col;
        output +=
            ((row + col) % 2 == 0) ? kBackgroundWhite : kBackgroundMagenta;
        output += kForegroundBlack;
        auto [pt, is_white] = board.at(sq);
        output += kGlyphs.table[is_white ? 0 : 1][static_cast<size_t>(pt)];
      } else {
        output += file;
      }
      output += ' ';
    }
    output += kBackgroundDefault;
    output += kForegroundGray;
    output += ' ';
    for (auto move = static_cast<std::size_t>(row); move < history.row_count();
         move += 9) {
      output += history.row_str(move);
    }
    output += kForegroundDefault;
    output += '\n';
  }
  return output;
}

}  // namespace

std::function<Score(const ChessMove&)> black_advantage_on_capture =
    [](const ChessMove& move) {
      constexpr std::array<Score, 7> kMaterialValues = {0, 200, 9, 5, 3, 3, 1};
      return kMaterialValues[static_cast<size_t>(move.captured)];
    };

int main() {
  auto game = ChessState::initial_position();
  MoveHistory history;
  int fullmove_number = 1;

  std::vector<std::unique_ptr<PlayerBase<ChessMove>>> agents;
  agents.push_back(std::make_unique<HumanPlayer<ChessState>>(game));
  agents.push_back(std::make_unique<MinimaxPlayer<ChessState>>(
      game, 5, black_advantage_on_capture));

  while (true) {
    for (const auto& agent_ptr : agents) {
      std::cout << RenderGame(game, history) << "\n";
      if (game.IsOver()) {
        return 0;
      }
      const auto move = agent_ptr->SelectMove();
      history.Record(fullmove_number, to_san(move));
      ChessState::MoveUndo undo;
      game.MakeMove(move, undo);
      if (game.IsOver()) {
        std::cout << RenderGame(game, history) << "\n";
        return 0;
      }
      // Black just moved; advance the fullmove counter for the next White move.
      if (agent_ptr == agents[1]) {
        ++fullmove_number;
      }
    }
  }
}
