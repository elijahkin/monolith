#include "chess_state.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>
#include <string>
#include <vector>

ChessState::AttackTables ChessState::compute_attack_tables() {
  AttackTables a{};
  constexpr int kd[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                            {1, -2},  {1, 2},  {2, -1},  {2, 1}};
  constexpr int kKingDeltas[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                     {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  for (int sq = 0; sq < 64; ++sq) {
    int r = sq / 8, f = sq % 8;
    for (const auto& d : kd) {
      int nr = r + d[0], nf = f + d[1];
      if (0 <= nr && nr < 8 && 0 <= nf && nf < 8)
        a.knight[sq] |= square_bb((nr * 8) + nf);
    }
    for (const auto& d : kKingDeltas) {
      int nr = r + d[0], nf = f + d[1];
      if (0 <= nr && nr < 8 && 0 <= nf && nf < 8)
        a.king[sq] |= square_bb((nr * 8) + nf);
    }
  }
  return a;
}

ChessState::Bitboard ChessState::sliding_targets(Bitboard occ, int sq,
                                                 int dir) const {
  Bitboard targets = 0;
  int step = dir;
  for (int s = sq + step; 0 <= s && s < 64; s += step) {
    int df = (s % 8) - ((s - step) % 8);
    if (df > 1 || df < -1) {
      break;
    }
    targets |= square_bb(s);
    if (occ & square_bb(s)) {
      break;
    }
  }
  return targets;
}

bool ChessState::is_attacked(int sq, Color by) const {
  if (by == Color::kWhite) {
    Bitboard pawns =
        pieces_[static_cast<size_t>(PieceType::kPawn)] & colors_[0];
    if (sq >= 8) {
      if ((sq % 8) > 0 && (pawns & square_bb(sq - 9))) return true;
      if ((sq % 8) < 7 && (pawns & square_bb(sq - 7))) return true;
    }
  } else {
    Bitboard pawns =
        pieces_[static_cast<size_t>(PieceType::kPawn)] & colors_[1];
    if (sq < 56) {
      if ((sq % 8) > 0 && (pawns & square_bb(sq + 7))) return true;
      if ((sq % 8) < 7 && (pawns & square_bb(sq + 9))) return true;
    }
  }

  Bitboard enemy = colors_[static_cast<size_t>(by)];
  if (kAttacks.knight[sq] & pieces_[static_cast<size_t>(PieceType::kKnight)] &
      enemy)
    return true;
  if (kAttacks.king[sq] & pieces_[static_cast<size_t>(PieceType::kKing)] &
      enemy)
    return true;
  if (bishop_targets(sq, occupied()) &
      (pieces_[static_cast<size_t>(PieceType::kBishop)] |
       pieces_[static_cast<size_t>(PieceType::kQueen)]) &
      enemy)
    return true;
  if (rook_targets(sq, occupied()) &
      (pieces_[static_cast<size_t>(PieceType::kRook)] |
       pieces_[static_cast<size_t>(PieceType::kQueen)]) &
      enemy)
    return true;
  return false;
}

void ChessState::add_pawn_moves(std::vector<Move>& moves) const {
  Bitboard pawns = pieces_[static_cast<size_t>(PieceType::kPawn)] & us();
  Bitboard occ = occupied();
  Bitboard them_bb = them();
  bool is_white = side_to_move_ == Color::kWhite;
  int push_dir = is_white ? 8 : -8;
  int start_rank = is_white ? 1 : 6;
  int promo_rank = is_white ? 6 : 1;

  while (pawns) {
    int sq = std::countr_zero(pawns);
    pawns &= pawns - 1;
    int r = sq / 8;

    int push_sq = sq + push_dir;
    if (!(occ & square_bb(push_sq))) {
      if (r == promo_rank) {
        for (auto pt : {PieceType::kKnight, PieceType::kBishop,
                        PieceType::kRook, PieceType::kQueen}) {
          moves.push_back({sq, push_sq, pt});
        }
      } else {
        moves.push_back({sq, push_sq, PieceType::kNone});
      }
      if (r == start_rank) {
        int dp_sq = sq + (2 * push_dir);
        if (!(occ & square_bb(dp_sq))) {
          moves.push_back({sq, dp_sq, PieceType::kNone});
        }
      }
    }

    int cap_sqs[2];
    int n_caps = 0;
    if (sq % 8 > 0) {
      cap_sqs[n_caps++] = sq + push_dir - 1;
    }
    if (sq % 8 < 7) {
      cap_sqs[n_caps++] = sq + push_dir + 1;
    }

    for (int i = 0; i < n_caps; ++i) {
      int cap_sq = cap_sqs[i];
      if (square_bb(cap_sq) & them_bb) {
        if (r == promo_rank) {
          for (auto pt : {PieceType::kKnight, PieceType::kBishop,
                          PieceType::kRook, PieceType::kQueen})
            moves.push_back({sq, cap_sq, pt});
        } else {
          moves.push_back({sq, cap_sq, PieceType::kNone});
        }
      }
      if (cap_sq == en_passant_square_)
        moves.push_back({sq, cap_sq, PieceType::kNone});
    }
  }
}

void ChessState::add_knight_moves(std::vector<Move>& moves) const {
  add_piece_moves(moves,
                  pieces_[static_cast<size_t>(PieceType::kKnight)] & us(),
                  [this](int sq) { return kAttacks.knight[sq]; });
}

void ChessState::add_king_moves(std::vector<Move>& moves) const {
  int sq =
      std::countr_zero(pieces_[static_cast<size_t>(PieceType::kKing)] & us());
  Bitboard targets = kAttacks.king[sq] & ~us();
  while (targets) {
    int t = std::countr_zero(targets);
    targets &= targets - 1;
    moves.push_back({sq, t, PieceType::kNone});
  }

  Bitboard occ = occupied();
  if (side_to_move_ == Color::kWhite) {
    if ((castling_rights_ & 1) && !(occ & 0x60) &&
        !is_attacked(4, Color::kBlack) && !is_attacked(5, Color::kBlack)) {
      moves.push_back({square(File::E, Rank::R1), square(File::G, Rank::R1),
                       PieceType::kNone});
    }
    if ((castling_rights_ & 2) && !(occ & 0x0E) &&
        !is_attacked(4, Color::kBlack) && !is_attacked(3, Color::kBlack)) {
      moves.push_back({square(File::E, Rank::R1), square(File::C, Rank::R1),
                       PieceType::kNone});
    }
  } else {
    if ((castling_rights_ & 4) && !(occ & 0x6000000000000000) &&
        !is_attacked(60, Color::kWhite) && !is_attacked(61, Color::kWhite)) {
      moves.push_back({square(File::E, Rank::R8), square(File::G, Rank::R8),
                       PieceType::kNone});
    }
    if ((castling_rights_ & 8) && !(occ & 0x0E00000000000000) &&
        !is_attacked(60, Color::kWhite) && !is_attacked(59, Color::kWhite)) {
      moves.push_back({square(File::E, Rank::R8), square(File::C, Rank::R8),
                       PieceType::kNone});
    }
  }
}

void ChessState::add_bishop_moves(std::vector<Move>& moves) const {
  Bitboard occ = occupied();
  add_piece_moves(moves,
                  pieces_[static_cast<size_t>(PieceType::kBishop)] & us(),
                  [this, occ](int sq) { return bishop_targets(sq, occ); });
}

void ChessState::add_rook_moves(std::vector<Move>& moves) const {
  Bitboard occ = occupied();
  add_piece_moves(moves, pieces_[static_cast<size_t>(PieceType::kRook)] & us(),
                  [this, occ](int sq) { return rook_targets(sq, occ); });
}

void ChessState::add_queen_moves(std::vector<Move>& moves) const {
  Bitboard occ = occupied();
  add_piece_moves(moves, pieces_[static_cast<size_t>(PieceType::kQueen)] & us(),
                  [this, occ](int sq) { return queen_targets(sq, occ); });
}

std::vector<ChessState::Move> ChessState::pseudo_legal_moves() const {
  std::vector<Move> moves;
  add_pawn_moves(moves);
  add_knight_moves(moves);
  add_bishop_moves(moves);
  add_rook_moves(moves);
  add_queen_moves(moves);
  add_king_moves(moves);
  return moves;
}

double ChessState::evaluate() const {
  if (in_check() && legal_moves().empty()) return -1e6;
  constexpr double kPieceValues[] = {0, 1, 3, 3, 5, 9, 0};
  double score = 0;
  for (int pt = 1; pt <= 5; ++pt) {
    int cnt = std::popcount(pieces_[static_cast<size_t>(pt)]);
    if (side_to_move_ == Color::kWhite) {
      score += kPieceValues[pt] * cnt;
    } else {
      score -= kPieceValues[pt] * cnt;
    }
  }
  return score;
}
std::vector<ChessState::Move> ChessState::legal_moves() const {
  std::vector<Move> candidates = pseudo_legal_moves();
  std::vector<Move> result;
  Color us = side_to_move_;

  for (auto& m : candidates) {
    MoveUndo undo = const_cast<ChessState*>(this)->make_move(m);
    int king_sq =
        std::countr_zero(pieces_[static_cast<size_t>(PieceType::kKing)] &
                         colors_[static_cast<size_t>(us)]);
    bool legal =
        !is_attacked(king_sq, static_cast<Color>(1 - static_cast<size_t>(us)));
    const_cast<ChessState*>(this)->unmake_move(m, undo);

    if (legal) {
      result.push_back(m);
    }
  }
  return result;
}

ChessState::MoveUndo ChessState::make_move(Move m) {
  bool is_white = side_to_move_ == Color::kWhite;
  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  MoveUndo undo;
  undo.old_castling_rights = castling_rights_;
  undo.old_en_passant_square = en_passant_square_;
  undo.old_halfmove_clock = halfmove_clock_;
  undo.captured_piece = PieceType::kNone;

  PieceType pt = PieceType::kNone;
  for (int i = 1; i <= 6; ++i) {
    if (pieces_[static_cast<size_t>(i)] & from_bb) {
      pt = static_cast<PieceType>(i);
      break;
    }
  }

  for (int i = 1; i <= 5; ++i) {
    if (pieces_[static_cast<size_t>(i)] & to_bb & them()) {
      undo.captured_piece = static_cast<PieceType>(i);
      pieces_[static_cast<size_t>(i)] ^= to_bb;
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= to_bb;
      break;
    }
  }

  if (pt == PieceType::kPawn && m.to == en_passant_square_) {
    int ep_pawn_sq = en_passant_square_ + (is_white ? -8 : 8);
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
    colors_[1 - static_cast<size_t>(side_to_move_)] ^= square_bb(ep_pawn_sq);
    undo.captured_piece = PieceType::kPawn;
  }

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[static_cast<size_t>(side_to_move_)] ^= from_bb | to_bb;

  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
  }

  if (pt == PieceType::kKing) {
    if (m.from == square(File::E, Rank::R1) &&
        m.to == square(File::G, Rank::R1)) {
      Bitboard rook = square_bb(square(File::H, Rank::R1));
      Bitboard rook_to = square_bb(square(File::F, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
    } else if (m.from == square(File::E, Rank::R1) &&
               m.to == square(File::C, Rank::R1)) {
      Bitboard rook = square_bb(square(File::A, Rank::R1));
      Bitboard rook_to = square_bb(square(File::D, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::G, Rank::R8)) {
      Bitboard rook = square_bb(square(File::H, Rank::R8));
      Bitboard rook_to = square_bb(square(File::F, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::C, Rank::R8)) {
      Bitboard rook = square_bb(square(File::A, Rank::R8));
      Bitboard rook_to = square_bb(square(File::D, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
    }
  }

  uint8_t clear_mask = 0;
  if (m.from == 0 || m.to == 0) clear_mask |= 2;
  if (m.from == 7 || m.to == 7) clear_mask |= 1;
  if (m.from == 56 || m.to == 56) clear_mask |= 8;
  if (m.from == 63 || m.to == 63) clear_mask |= 4;
  if (m.from == 4) clear_mask |= 3;
  if (m.from == 60) clear_mask |= 12;
  castling_rights_ &= ~clear_mask;

  if (pt == PieceType::kPawn && (m.to - m.from == 16 || m.to - m.from == -16)) {
    en_passant_square_ = (m.from + m.to) / 2;
  } else {
    en_passant_square_ = -1;
  }

  if (pt == PieceType::kPawn || undo.captured_piece != PieceType::kNone) {
    halfmove_clock_ = 0;
  } else {
    ++halfmove_clock_;
  }

  if (is_white) {
    ++fullmove_number_;
  }
  side_to_move_ = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));

  return undo;
}

void ChessState::unmake_move(Move m, const MoveUndo& undo) {
  bool is_white = side_to_move_ == Color::kBlack;
  side_to_move_ = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));

  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  PieceType pt = PieceType::kNone;
  for (int i = 1; i <= 6; ++i) {
    if (pieces_[static_cast<size_t>(i)] & to_bb) {
      if (m.promotion != PieceType::kNone &&
          i == static_cast<int>(m.promotion)) {
        pt = PieceType::kPawn;
        break;
      } else if (m.promotion == PieceType::kNone) {
        pt = static_cast<PieceType>(i);
        break;
      }
    }
  }

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[static_cast<size_t>(side_to_move_)] ^= from_bb | to_bb;

  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
  }

  if (undo.captured_piece != PieceType::kNone) {
    if (pt == PieceType::kPawn && m.to == undo.old_en_passant_square) {
      int ep_pawn_sq = undo.old_en_passant_square + (is_white ? -8 : 8);
      pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= square_bb(ep_pawn_sq);
    } else {
      pieces_[static_cast<size_t>(undo.captured_piece)] ^= to_bb;
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= to_bb;
    }
  }

  if (pt == PieceType::kKing) {
    if (m.from == square(File::E, Rank::R1) &&
        m.to == square(File::G, Rank::R1)) {
      Bitboard rook = square_bb(square(File::H, Rank::R1));
      Bitboard rook_to = square_bb(square(File::F, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
    } else if (m.from == square(File::E, Rank::R1) &&
               m.to == square(File::C, Rank::R1)) {
      Bitboard rook = square_bb(square(File::A, Rank::R1));
      Bitboard rook_to = square_bb(square(File::D, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::G, Rank::R8)) {
      Bitboard rook = square_bb(square(File::H, Rank::R8));
      Bitboard rook_to = square_bb(square(File::F, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::C, Rank::R8)) {
      Bitboard rook = square_bb(square(File::A, Rank::R8));
      Bitboard rook_to = square_bb(square(File::D, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
    }
  }

  castling_rights_ = undo.old_castling_rights;
  en_passant_square_ = undo.old_en_passant_square;
  halfmove_clock_ = undo.old_halfmove_clock;
  if (!is_white) {
    --fullmove_number_;
  }
}

ChessState ChessState::initial_position() {
  ChessState s;
  s.colors_[0] = 0x000000000000FFFFUL;
  s.colors_[1] = 0xFFFF000000000000UL;
  s.pieces_[static_cast<size_t>(PieceType::kPawn)] = 0x00FF00000000FF00UL;
  s.pieces_[static_cast<size_t>(PieceType::kKnight)] = 0x4200000000000042UL;
  s.pieces_[static_cast<size_t>(PieceType::kBishop)] = 0x2400000000000024UL;
  s.pieces_[static_cast<size_t>(PieceType::kRook)] = 0x8100000000000081UL;
  s.pieces_[static_cast<size_t>(PieceType::kQueen)] = 0x0800000000000008UL;
  s.pieces_[static_cast<size_t>(PieceType::kKing)] = 0x1000000000000010UL;
  return s;
}

PieceType ChessState::piece_type_at(int sq) const {
  Bitboard bb = square_bb(sq);
  for (int i = 1; i <= 6; ++i) {
    if (pieces_[static_cast<size_t>(i)] & bb) {
      return static_cast<PieceType>(i);
    }
  }
  return PieceType::kNone;
}

int ChessState::unicode_index(int sq) const {
  Bitboard bb = square_bb(sq);
  for (int i = 1; i <= 6; ++i) {
    if (pieces_[static_cast<size_t>(i)] & bb) {
      if (colors_[0] & bb) return i;
      return i + 6;
    }
  }
  return 0;
}

std::string ChessState::get_algebraic_notation(const ChessMove& move) const {
  static constexpr std::array<char, 7> kPieceLetters = {'\0', '\0', 'N', 'B',
                                                        'R',  'Q',  'K'};
  std::string output;
  output += kPieceLetters[static_cast<size_t>(piece_type_at(move.from))];
  if (move.captured != PieceType::kNone) {
    output += 'x';
  }
  output += static_cast<char>('a' + (move.to % 8));
  output += std::to_string(1 + (move.to / 8));
  return output;
}

void ChessState::RecordMove(const ChessMove& move) {
  if (side_to_move_ == Color::kWhite) {
    history_.push_back(get_algebraic_notation(move));
  } else {
    std::string& tmp = history_.back();
    tmp.insert(tmp.length(), " ");
    tmp.insert(tmp.length(), get_algebraic_notation(move));
  }
}

void ChessState::MakeMove(const ChessMove& move) {
  Move m{move.from, move.to, move.promotion};
  undo_stack_.push_back(make_move(m));
}

void ChessState::UnmakeMove(const ChessMove& move) {
  Move m{move.from, move.to, move.promotion};
  unmake_move(m, undo_stack_.back());
  undo_stack_.pop_back();
}

std::vector<ChessMove> ChessState::GenerateLegalMoves() const {
  auto moves = legal_moves();
  std::vector<ChessMove> result;
  Bitboard them_bb = them();
  for (const auto& m : moves) {
    ChessMove cm;
    cm.from = m.from;
    cm.to = m.to;
    cm.promotion = m.promotion;
    cm.captured = PieceType::kNone;
    for (int i = 1; i <= 5; ++i) {
      if (pieces_[static_cast<size_t>(i)] & square_bb(m.to) & them_bb) {
        cm.captured = static_cast<PieceType>(i);
        break;
      }
    }
    if (cm.captured == PieceType::kNone && m.promotion == PieceType::kNone &&
        m.to == en_passant_square_) {
      cm.captured = PieceType::kPawn;
    }
    result.push_back(cm);
  }
  return result;
}

std::string ChessState::ToString() const {
  static constexpr std::array<const char*, 13> kUnicodePieces = {
      " ",      "\u2654", "\u2655", "\u2656", "\u2657", "\u2658", "\u2659",
      "\u265a", "\u265b", "\u265c", "\u265d", "\u265e", "\u265f"};

  const std::string kCursorHome = "\x1B[H";
  const std::string kEraseScreen = "\x1B[2J";
  const std::string kForegroundBlack = "\x1B[30m";
  const std::string kForegroundGray = "\x1B[38;5;240m";
  const std::string kForegroundDefault = "\x1B[39m";
  const std::string kBackgroundMagenta = "\x1B[45m";
  const std::string kBackgroundWhite = "\x1B[47m";
  const std::string kBackgroundDefault = "\x1B[49m";

  std::string output = kEraseScreen + kCursorHome;
  for (int row = 0; row < 9; ++row) {
    char rank = static_cast<char>('8' - row);
    output += (row == 8 ? ' ' : rank);
    output += ' ';
    for (int col = 0; col < 8; ++col) {
      char file = static_cast<char>('a' + col);
      if (row != 8) {
        int sq = (8 * (7 - row)) + col;
        output +=
            ((row + col) % 2 == 0) ? kBackgroundWhite : kBackgroundMagenta;
        output += kForegroundBlack;
        output += kUnicodePieces[unicode_index(sq)];
      } else {
        output += file;
      }
      output += ' ';
    }
    output += kBackgroundDefault;
    output += kForegroundGray;
    output += ' ';
    for (size_t move = static_cast<size_t>(row); move < history_.size();
         move += 9) {
      std::string move_str = std::to_string(move + 1);
      move_str.insert(0, 3 - move_str.size(), ' ');
      move_str += ". ";
      move_str += history_[move];
      move_str.insert(move_str.end(), 14 - move_str.size(), ' ');
      output += move_str;
    }
    output += kForegroundDefault;
    output += '\n';
  }
  return output;
}

std::optional<ChessMove> ChessState::Parse(const std::string& input) const {
  const std::regex san_pattern("([KQRBN]?)x?([a-h])([1-8])");

  std::smatch san;
  if (!std::regex_match(input, san, san_pattern)) {
    return std::nullopt;
  }

  PieceType type = PieceType::kNone;
  switch (san[1].str()[0]) {
    case 'K':
      type = PieceType::kKing;
      break;
    case 'Q':
      type = PieceType::kQueen;
      break;
    case 'R':
      type = PieceType::kRook;
      break;
    case 'B':
      type = PieceType::kBishop;
      break;
    case 'N':
      type = PieceType::kKnight;
      break;
    default:
      type = PieceType::kPawn;
      break;
  }
  int to_file = san[2].str()[0] - 'a';
  int to_rank = san[3].str()[0] - '1';
  int to = (8 * to_rank) + to_file;

  auto moves = GenerateLegalMoves();
  std::vector<ChessMove> candidates;
  for (const auto& move : moves) {
    if (move.to == to && piece_type_at(move.from) == type) {
      candidates.push_back(move);
    }
  }
  if (candidates.size() != 1) {
    return std::nullopt;
  }
  return candidates[0];
}
