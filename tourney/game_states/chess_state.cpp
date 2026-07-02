#include "chess_state.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
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

  // If the move puts us in check, it's not legal
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

  // Store undo information
  MoveUndo undo;
  undo.old_castling_rights = castling_rights_;
  undo.old_en_passant_square = en_passant_square_;
  undo.old_halfmove_clock = halfmove_clock_;
  undo.captured_piece = PieceType::kNone;

  // Find the piece being moved
  PieceType pt = PieceType::kNone;
  for (int i = 1; i <= 6; ++i) {
    if (pieces_[static_cast<size_t>(i)] & from_bb) {
      pt = static_cast<PieceType>(i);
      break;
    }
  }

  // Handle captures
  for (int i = 1; i <= 5; ++i) {
    if (pieces_[static_cast<size_t>(i)] & to_bb & them()) {
      undo.captured_piece = static_cast<PieceType>(i);
      pieces_[static_cast<size_t>(i)] ^= to_bb;
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= to_bb;
      break;
    }
  }

  // Handle en passant
  if (pt == PieceType::kPawn && m.to == en_passant_square_) {
    int ep_pawn_sq = en_passant_square_ + (is_white ? -8 : 8);
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
    colors_[1 - static_cast<size_t>(side_to_move_)] ^= square_bb(ep_pawn_sq);
    undo.captured_piece = PieceType::kPawn;
  }

  // Move the piece
  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[static_cast<size_t>(side_to_move_)] ^= from_bb | to_bb;

  // Handle promotion
  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
  }

  // Handle castling (move rook)
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

  // Update castling rights
  uint8_t clear_mask = 0;
  if (m.from == 0 || m.to == 0) clear_mask |= 2;
  if (m.from == 7 || m.to == 7) clear_mask |= 1;
  if (m.from == 56 || m.to == 56) clear_mask |= 8;
  if (m.from == 63 || m.to == 63) clear_mask |= 4;
  if (m.from == 4) clear_mask |= 3;
  if (m.from == 60) clear_mask |= 12;
  castling_rights_ &= ~clear_mask;

  // Update en passant square
  if (pt == PieceType::kPawn && (m.to - m.from == 16 || m.to - m.from == -16)) {
    en_passant_square_ = (m.from + m.to) / 2;
  } else {
    en_passant_square_ = -1;
  }

  // Update halfmove clock
  if (pt == PieceType::kPawn || undo.captured_piece != PieceType::kNone) {
    halfmove_clock_ = 0;
  } else {
    ++halfmove_clock_;
  }

  // Update side to move and fullmove number
  if (is_white) {
    ++fullmove_number_;
  }
  side_to_move_ = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));

  return undo;
}

void ChessState::unmake_move(Move m, const MoveUndo& undo) {
  // Toggle side to move first
  bool is_white = side_to_move_ == Color::kBlack;  // Opposite after toggle
  side_to_move_ = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));

  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  // Find the piece that was moved
  PieceType pt = PieceType::kNone;
  for (int i = 1; i <= 6; ++i) {
    if (pieces_[static_cast<size_t>(i)] & to_bb) {
      // After promotion, to_bb has the promoting piece type
      if (m.promotion != PieceType::kNone &&
          i == static_cast<int>(m.promotion)) {
        pt = PieceType::kPawn;  // Was originally a pawn
        break;
      } else if (m.promotion == PieceType::kNone) {
        pt = static_cast<PieceType>(i);
        break;
      }
    }
  }

  // Reverse the piece move
  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[static_cast<size_t>(side_to_move_)] ^= from_bb | to_bb;

  // Handle promotion reversal
  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
  }

  // Restore captured piece
  if (undo.captured_piece != PieceType::kNone) {
    if (pt == PieceType::kPawn && m.to == undo.old_en_passant_square) {
      // En passant: restore pawn to its original square
      int ep_pawn_sq = undo.old_en_passant_square + (is_white ? -8 : 8);
      pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= square_bb(ep_pawn_sq);
    } else {
      // Normal capture: restore piece to destination square
      pieces_[static_cast<size_t>(undo.captured_piece)] ^= to_bb;
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= to_bb;
    }
  }

  // Reverse castling (unmove rook)
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

  // Restore all state
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
