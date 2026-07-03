#include "chess_state.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>
#include <string>
#include <vector>

// --- Magic bitboard table data ---
ChessState::Bitboard ChessState::rook_masks_[64];
int ChessState::rook_offset_[64];
ChessState::Bitboard* ChessState::rook_table_;

ChessState::Bitboard ChessState::bishop_masks_[64];
int ChessState::bishop_offset_[64];
ChessState::Bitboard* ChessState::bishop_table_;

bool ChessState::magic_initialized_ = false;

namespace {
using BB = uint64_t;

[[nodiscard]] BB sliding_targets(BB occ, int sq, int dir) {
  BB targets = 0;
  for (int s = sq + dir; 0 <= s && s < 64; s += dir) {
    int df = (s % 8) - ((s - dir) % 8);
    if (df > 1 || df < -1) break;
    targets |= (BB{1} << s);
    if (occ & (BB{1} << s)) break;
  }
  return targets;
}

[[nodiscard]] BB relevant_mask(int sq, bool bishop) {
  BB mask = 0;
  int dirs[4];
  if (bishop) {
    dirs[0] = 9;
    dirs[1] = -9;
    dirs[2] = 7;
    dirs[3] = -7;
  } else {
    dirs[0] = 8;
    dirs[1] = -8;
    dirs[2] = 1;
    dirs[3] = -1;
  }
  for (int di = 0; di < 4; ++di) {
    int dir = dirs[di];
    for (int s = sq + dir; 0 <= s && s < 64; s += dir) {
      int prev = s - dir;
      if ((s % 8) - (prev % 8) > 1 || (prev % 8) - (s % 8) > 1) break;
      int next = s + dir;
      bool terminal = !(0 <= next && next < 64) || ((next % 8) - (s % 8) > 1) ||
                      ((s % 8) - (next % 8) > 1);
      if (terminal) break;
      mask |= (BB{1} << s);
    }
  }
  return mask;
}

[[nodiscard]] BB expand(int idx, BB mask) {
  BB result = 0;
  while (mask) {
    int bit = std::countr_zero(mask);
    mask &= mask - 1;
    if (idx & 1) result |= (BB{1} << bit);
    idx >>= 1;
  }
  return result;
}

[[nodiscard]] BB attacks_for_occ(BB occ, int sq, bool bishop) {
  BB attacks = 0;
  int dirs[4];
  if (bishop) {
    dirs[0] = 9;
    dirs[1] = -9;
    dirs[2] = 7;
    dirs[3] = -7;
  } else {
    dirs[0] = 8;
    dirs[1] = -8;
    dirs[2] = 1;
    dirs[3] = -1;
  }
  for (int di = 0; di < 4; ++di) attacks |= sliding_targets(occ, sq, dirs[di]);
  return attacks;
}

}  // anonymous namespace

void ChessState::init_magic_table(bool bishop) {
  int total = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard mask = relevant_mask(sq, bishop);
    int bits = std::popcount(mask);
    int n = 1 << bits;
    if (bishop)
      bishop_offset_[sq] = total;
    else
      rook_offset_[sq] = total;
    total += n;
    if (bishop)
      bishop_masks_[sq] = mask;
    else
      rook_masks_[sq] = mask;
  }
  auto* table = new Bitboard[total];
  if (bishop)
    bishop_table_ = table;
  else
    rook_table_ = table;
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard mask = bishop ? bishop_masks_[sq] : rook_masks_[sq];
    int bits = std::popcount(mask);
    int n = 1 << bits;
    int offset = bishop ? bishop_offset_[sq] : rook_offset_[sq];
    for (int i = 0; i < n; ++i) {
      Bitboard occ = expand(i, mask);
      table[offset + i] = attacks_for_occ(occ, sq, bishop);
    }
  }
}

void ChessState::init_magic_tables() {
  if (magic_initialized_) return;
  magic_initialized_ = true;
  init_magic_table(false);
  init_magic_table(true);
}

// --- End magic bitboard table data ---

ChessState::AttackTables ChessState::compute_attack_tables() {
  init_magic_tables();
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

    if (f > 0 && r < 7) a.pawn_attacks[0][sq] |= square_bb(sq + 7);
    if (f < 7 && r < 7) a.pawn_attacks[0][sq] |= square_bb(sq + 9);
    if (f < 7 && r > 0) a.pawn_attacks[1][sq] |= square_bb(sq - 7);
    if (f > 0 && r > 0) a.pawn_attacks[1][sq] |= square_bb(sq - 9);
  }

  for (int sq1 = 0; sq1 < 64; ++sq1) {
    int f1 = sq1 & 7, r1 = sq1 >> 3;
    for (int sq2 = 0; sq2 < 64; ++sq2) {
      int f2 = sq2 & 7, r2 = sq2 >> 3;
      if (f1 == f2 || r1 == r2) {
        Bitboard occ = square_bb(sq1) | square_bb(sq2);
        a.between[sq1][sq2] = rook_targets(sq1, occ) & rook_targets(sq2, occ);
      } else {
        int df = f1 > f2 ? f1 - f2 : f2 - f1;
        int dr = r1 > r2 ? r1 - r2 : r2 - r1;
        if (df == dr) {
          Bitboard occ = square_bb(sq1) | square_bb(sq2);
          a.between[sq1][sq2] =
              bishop_targets(sq1, occ) & bishop_targets(sq2, occ);
        }
      }
    }
  }

  return a;
}

ChessState::Bitboard ChessState::compute_checkers(int king_sq,
                                                  Color them) const {
  Bitboard them_bb = colors_[static_cast<size_t>(them)];
  Bitboard occ = occupied();

  Bitboard knight_checkers = kAttacks.knight[king_sq] &
                             pieces_[static_cast<size_t>(PieceType::kKnight)] &
                             them_bb;

  Bitboard pawn_checkers =
      kAttacks.pawn_attacks[1 - static_cast<size_t>(them)][king_sq] &
      pieces_[static_cast<size_t>(PieceType::kPawn)] & them_bb;

  Bitboard bishop_queen = (pieces_[static_cast<size_t>(PieceType::kBishop)] |
                           pieces_[static_cast<size_t>(PieceType::kQueen)]) &
                          them_bb;
  Bitboard rook_queen = (pieces_[static_cast<size_t>(PieceType::kRook)] |
                         pieces_[static_cast<size_t>(PieceType::kQueen)]) &
                        them_bb;

  Bitboard bishop_checkers = bishop_targets(king_sq, occ) & bishop_queen;
  Bitboard rook_checkers = rook_targets(king_sq, occ) & rook_queen;

  return knight_checkers | pawn_checkers | bishop_checkers | rook_checkers;
}

ChessState::Bitboard ChessState::compute_pinned(int king_sq, Color us,
                                                Color them,
                                                Bitboard* pin_rays) const {
  Bitboard us_bb = colors_[static_cast<size_t>(us)];
  Bitboard them_bb = colors_[static_cast<size_t>(them)];

  Bitboard enemy_rooks = (pieces_[static_cast<size_t>(PieceType::kRook)] |
                          pieces_[static_cast<size_t>(PieceType::kQueen)]) &
                         them_bb;
  Bitboard enemy_bishops = (pieces_[static_cast<size_t>(PieceType::kBishop)] |
                            pieces_[static_cast<size_t>(PieceType::kQueen)]) &
                           them_bb;

  if (!(rook_targets(king_sq, 0) & enemy_rooks) &&
      !(bishop_targets(king_sq, 0) & enemy_bishops))
    return 0;

  Bitboard rook_pinners = rook_targets(king_sq, them_bb) & enemy_rooks;
  Bitboard bishop_pinners = bishop_targets(king_sq, them_bb) & enemy_bishops;

  Bitboard pinners = rook_pinners | bishop_pinners;
  Bitboard pinned = 0;

  while (pinners) {
    int psq = std::countr_zero(pinners);
    pinners &= pinners - 1;
    Bitboard between = between_bb(king_sq, psq);
    Bitboard friends = between & us_bb;
    if (friends && !(friends & (friends - 1))) {
      int fsq = std::countr_zero(friends);
      pinned |= square_bb(fsq);
      if (pin_rays) {
        pin_rays[fsq] = between | square_bb(psq);
      }
    }
  }
  return pinned;
}

void ChessState::add_pawn_moves(std::vector<ChessMove>& moves) const {
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

    Bitboard cap_targets = kAttacks.pawn_attacks[!is_white][sq] & them_bb;
    while (cap_targets) {
      int cap_sq = std::countr_zero(cap_targets);
      cap_targets &= cap_targets - 1;
      if (r == promo_rank) {
        for (auto pt : {PieceType::kKnight, PieceType::kBishop,
                        PieceType::kRook, PieceType::kQueen})
          moves.push_back({sq, cap_sq, pt});
      } else {
        moves.push_back({sq, cap_sq, PieceType::kNone});
      }
    }

    Bitboard ep_cap =
        kAttacks.pawn_attacks[!is_white][sq] & square_bb(en_passant_square_);
    if (ep_cap) moves.push_back({sq, en_passant_square_, PieceType::kNone});
  }
}

void ChessState::add_knight_moves(std::vector<ChessMove>& moves) const {
  add_piece_moves(moves,
                  pieces_[static_cast<size_t>(PieceType::kKnight)] & us(),
                  [this](int sq) { return kAttacks.knight[sq]; });
}

void ChessState::add_king_moves(std::vector<ChessMove>& moves) const {
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

void ChessState::add_bishop_moves(std::vector<ChessMove>& moves) const {
  Bitboard occ = occupied();
  add_piece_moves(moves,
                  pieces_[static_cast<size_t>(PieceType::kBishop)] & us(),
                  [this, occ](int sq) { return bishop_targets(sq, occ); });
}

void ChessState::add_rook_moves(std::vector<ChessMove>& moves) const {
  Bitboard occ = occupied();
  add_piece_moves(moves, pieces_[static_cast<size_t>(PieceType::kRook)] & us(),
                  [this, occ](int sq) { return rook_targets(sq, occ); });
}

void ChessState::add_queen_moves(std::vector<ChessMove>& moves) const {
  Bitboard occ = occupied();
  add_piece_moves(moves, pieces_[static_cast<size_t>(PieceType::kQueen)] & us(),
                  [this, occ](int sq) { return queen_targets(sq, occ); });
}

std::vector<ChessMove> ChessState::pseudo_legal_moves() const {
  std::vector<ChessMove> moves;
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

std::vector<ChessMove> ChessState::legal_moves() const {
  std::vector<ChessMove> candidates = pseudo_legal_moves();
  std::vector<ChessMove> result;
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

ChessState::MoveUndo ChessState::make_move(ChessMove m) {
  bool is_white = side_to_move_ == Color::kWhite;
  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  MoveUndo undo;
  undo.old_castling_rights = castling_rights_;
  undo.old_en_passant_square = en_passant_square_;
  undo.old_halfmove_clock = halfmove_clock_;
  undo.captured_piece = PieceType::kNone;

  PieceType pt = board_[m.from];

  if (board_[m.to] != PieceType::kNone) {
    undo.captured_piece = board_[m.to];
    pieces_[static_cast<size_t>(undo.captured_piece)] ^= to_bb;
    colors_[1 - static_cast<size_t>(side_to_move_)] ^= to_bb;
  }

  if (pt == PieceType::kPawn && m.to == en_passant_square_) {
    int ep_pawn_sq = en_passant_square_ + (is_white ? -8 : 8);
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
    colors_[1 - static_cast<size_t>(side_to_move_)] ^= square_bb(ep_pawn_sq);
    undo.captured_piece = PieceType::kPawn;
    board_[ep_pawn_sq] = PieceType::kNone;
  }

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[static_cast<size_t>(side_to_move_)] ^= from_bb | to_bb;

  board_[m.from] = PieceType::kNone;
  board_[m.to] = pt;

  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
    board_[m.to] = m.promotion;
  }

  if (pt == PieceType::kKing) {
    if (m.from == square(File::E, Rank::R1) &&
        m.to == square(File::G, Rank::R1)) {
      Bitboard rook = square_bb(square(File::H, Rank::R1));
      Bitboard rook_to = square_bb(square(File::F, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
      board_[square(File::H, Rank::R1)] = PieceType::kNone;
      board_[square(File::F, Rank::R1)] = PieceType::kRook;
    } else if (m.from == square(File::E, Rank::R1) &&
               m.to == square(File::C, Rank::R1)) {
      Bitboard rook = square_bb(square(File::A, Rank::R1));
      Bitboard rook_to = square_bb(square(File::D, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
      board_[square(File::A, Rank::R1)] = PieceType::kNone;
      board_[square(File::D, Rank::R1)] = PieceType::kRook;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::G, Rank::R8)) {
      Bitboard rook = square_bb(square(File::H, Rank::R8));
      Bitboard rook_to = square_bb(square(File::F, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
      board_[square(File::H, Rank::R8)] = PieceType::kNone;
      board_[square(File::F, Rank::R8)] = PieceType::kRook;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::C, Rank::R8)) {
      Bitboard rook = square_bb(square(File::A, Rank::R8));
      Bitboard rook_to = square_bb(square(File::D, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
      board_[square(File::A, Rank::R8)] = PieceType::kNone;
      board_[square(File::D, Rank::R8)] = PieceType::kRook;
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

void ChessState::unmake_move(ChessMove m, const MoveUndo& undo) {
  bool is_white = side_to_move_ == Color::kBlack;
  side_to_move_ = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));

  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  PieceType pt = board_[m.to];
  if (m.promotion != PieceType::kNone) pt = PieceType::kPawn;

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[static_cast<size_t>(side_to_move_)] ^= from_bb | to_bb;

  board_[m.from] = pt;
  board_[m.to] = PieceType::kNone;

  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
  }

  if (undo.captured_piece != PieceType::kNone) {
    if (pt == PieceType::kPawn && m.to == undo.old_en_passant_square) {
      int ep_pawn_sq = undo.old_en_passant_square + (is_white ? -8 : 8);
      pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= square_bb(ep_pawn_sq);
      board_[ep_pawn_sq] = PieceType::kPawn;
    } else {
      pieces_[static_cast<size_t>(undo.captured_piece)] ^= to_bb;
      colors_[1 - static_cast<size_t>(side_to_move_)] ^= to_bb;
      board_[m.to] = undo.captured_piece;
    }
  }

  if (pt == PieceType::kKing) {
    if (m.from == square(File::E, Rank::R1) &&
        m.to == square(File::G, Rank::R1)) {
      Bitboard rook = square_bb(square(File::H, Rank::R1));
      Bitboard rook_to = square_bb(square(File::F, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
      board_[square(File::H, Rank::R1)] = PieceType::kRook;
      board_[square(File::F, Rank::R1)] = PieceType::kNone;
    } else if (m.from == square(File::E, Rank::R1) &&
               m.to == square(File::C, Rank::R1)) {
      Bitboard rook = square_bb(square(File::A, Rank::R1));
      Bitboard rook_to = square_bb(square(File::D, Rank::R1));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[0] ^= rook | rook_to;
      board_[square(File::A, Rank::R1)] = PieceType::kRook;
      board_[square(File::D, Rank::R1)] = PieceType::kNone;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::G, Rank::R8)) {
      Bitboard rook = square_bb(square(File::H, Rank::R8));
      Bitboard rook_to = square_bb(square(File::F, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
      board_[square(File::H, Rank::R8)] = PieceType::kRook;
      board_[square(File::F, Rank::R8)] = PieceType::kNone;
    } else if (m.from == square(File::E, Rank::R8) &&
               m.to == square(File::C, Rank::R8)) {
      Bitboard rook = square_bb(square(File::A, Rank::R8));
      Bitboard rook_to = square_bb(square(File::D, Rank::R8));
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[1] ^= rook | rook_to;
      board_[square(File::A, Rank::R8)] = PieceType::kRook;
      board_[square(File::D, Rank::R8)] = PieceType::kNone;
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
  s.board_.fill(PieceType::kNone);
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard bb = Bitboard{1} << sq;
    for (int i = 1; i <= 6; ++i) {
      if (s.pieces_[static_cast<size_t>(i)] & bb) {
        s.board_[sq] = static_cast<PieceType>(i);
        break;
      }
    }
  }
  return s;
}

ChessState ChessState::from_fen(std::string_view fen) {
  ChessState s;
  s.board_.fill(PieceType::kNone);
  s.castling_rights_ = 0;
  s.en_passant_square_ = -1;
  s.halfmove_clock_ = 0;
  s.fullmove_number_ = 1;

  auto it = fen.begin();
  auto end = fen.end();

  for (int rank = 7; rank >= 0; --rank) {
    int file = 0;
    while (file < 8 && it != end) {
      char c = *it++;
      if (c >= '1' && c <= '8') {
        file += static_cast<int>(c - '0');
        continue;
      }
      PieceType pt;
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
          pt = PieceType::kNone;
          break;
      }
      bool is_white = (c >= 'A' && c <= 'Z');
      int sq = rank * 8 + file;
      s.board_[sq] = pt;
      s.pieces_[static_cast<size_t>(pt)] |= (Bitboard{1} << sq);
      s.colors_[is_white ? 0 : 1] |= (Bitboard{1} << sq);
      ++file;
    }
    if (it != end && *it == '/') ++it;
  }

  if (it != end && *it == ' ') ++it;

  if (it != end) s.side_to_move_ = (*it == 'w') ? Color::kWhite : Color::kBlack;
  ++it;
  if (it != end && *it == ' ') ++it;

  if (it != end && *it != '-') {
    while (it != end && *it != ' ') {
      switch (*it) {
        case 'K':
          s.castling_rights_ |= 1;
          break;
        case 'Q':
          s.castling_rights_ |= 2;
          break;
        case 'k':
          s.castling_rights_ |= 4;
          break;
        case 'q':
          s.castling_rights_ |= 8;
          break;
      }
      ++it;
    }
  } else {
    ++it;
  }
  if (it != end && *it == ' ') ++it;

  if (it != end && *it != '-') {
    int file = *it - 'a';
    ++it;
    int rank = *it - '1';
    ++it;
    s.en_passant_square_ = rank * 8 + file;
  } else {
    ++it;
  }
  if (it != end && *it == ' ') ++it;

  int halfmove = 0;
  while (it != end && *it >= '0' && *it <= '9') {
    halfmove = halfmove * 10 + (*it - '0');
    ++it;
  }
  s.halfmove_clock_ = halfmove;
  if (it != end && *it == ' ') ++it;

  int fullmove = 0;
  while (it != end && *it >= '0' && *it <= '9') {
    fullmove = fullmove * 10 + (*it - '0');
    ++it;
  }
  s.fullmove_number_ = fullmove;

  return s;
}

std::string ChessState::to_fen() const {
  std::string fen;

  for (int rank = 7; rank >= 0; --rank) {
    if (rank < 7) fen += '/';
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      int sq = rank * 8 + file;
      PieceType pt = board_[sq];
      if (pt == PieceType::kNone) {
        ++empty;
      } else {
        if (empty > 0) {
          fen += static_cast<char>('0' + empty);
          empty = 0;
        }
        bool is_white = (colors_[0] & (Bitboard{1} << sq)) != 0;
        char c;
        switch (pt) {
          case PieceType::kPawn:
            c = 'P';
            break;
          case PieceType::kKnight:
            c = 'N';
            break;
          case PieceType::kBishop:
            c = 'B';
            break;
          case PieceType::kRook:
            c = 'R';
            break;
          case PieceType::kQueen:
            c = 'Q';
            break;
          case PieceType::kKing:
            c = 'K';
            break;
          default:
            c = '?';
            break;
        }
        if (!is_white) c = static_cast<char>(c - 'A' + 'a');
        fen += c;
      }
    }
    if (empty > 0) fen += static_cast<char>('0' + empty);
  }

  fen += ' ';
  fen += (side_to_move_ == Color::kWhite) ? 'w' : 'b';

  fen += ' ';
  if (castling_rights_ == 0) {
    fen += '-';
  } else {
    if (castling_rights_ & 1) fen += 'K';
    if (castling_rights_ & 2) fen += 'Q';
    if (castling_rights_ & 4) fen += 'k';
    if (castling_rights_ & 8) fen += 'q';
  }

  fen += ' ';
  if (en_passant_square_ == -1) {
    fen += '-';
  } else {
    fen += static_cast<char>('a' + (en_passant_square_ % 8));
    fen += static_cast<char>('1' + (en_passant_square_ / 8));
  }

  fen += ' ';
  fen += std::to_string(halfmove_clock_);

  fen += ' ';
  fen += std::to_string(fullmove_number_);

  return fen;
}

PieceType ChessState::piece_type_at(int sq) const { return board_[sq]; }

int ChessState::unicode_index(int sq) const {
  PieceType pt = board_[sq];
  if (pt == PieceType::kNone) return 0;
  if (colors_[0] & square_bb(sq)) return static_cast<int>(pt);
  return static_cast<int>(pt) + 6;
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
  ChessMove m{move.from, move.to, move.promotion};
  undo_stack_.push_back(make_move(m));
}

void ChessState::UnmakeMove(const ChessMove& move) {
  ChessMove m{move.from, move.to, move.promotion};
  unmake_move(m, undo_stack_.back());
  undo_stack_.pop_back();
}

int ChessState::FillLegalMoves(ChessMove* buffer, int capacity) const {
  int n = legal_moves_fast(buffer);
  (void)capacity;
  for (int i = 0; i < n; ++i) {
    auto& m = buffer[i];
    m.captured = board_[m.to];
    if (m.captured == PieceType::kNone && m.promotion == PieceType::kNone &&
        m.to == en_passant_square_) {
      m.captured = PieceType::kPawn;
    }
  }
  return n;
}

bool ChessState::IsOver() const {
  ChessMove buf[1];
  return FillLegalMoves(buf, 1) == 0;
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

  ChessMove buf[ChessMove::kMaxMoves];
  int n = FillLegalMoves(buf, ChessMove::kMaxMoves);
  std::vector<ChessMove> candidates;
  for (int i = 0; i < n; ++i) {
    if (buf[i].to == to && piece_type_at(buf[i].from) == type) {
      candidates.push_back(buf[i]);
    }
  }
  if (candidates.size() != 1) {
    return std::nullopt;
  }
  return candidates[0];
}

// ---- Fast move generation (stack-allocated) ----

void ChessState::add_pawn_moves_fast(ChessMove*& out, Bitboard pinned_bb,
                                     const Bitboard* pin_rays,
                                     Bitboard target_mask) const {
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
    bool pinned = (pinned_bb & square_bb(sq)) != 0;
    Bitboard pin_ray = pinned && pin_rays ? pin_rays[sq] : ~Bitboard{0};

    int push_sq = sq + push_dir;
    Bitboard push_bb = square_bb(push_sq);
    if (!(occ & push_bb)) {
      if ((push_bb & target_mask) && (!pinned || (push_bb & pin_ray))) {
        if (r == promo_rank) {
          *out++ = {sq, push_sq, PieceType::kKnight, PieceType::kPawn};
          *out++ = {sq, push_sq, PieceType::kBishop, PieceType::kPawn};
          *out++ = {sq, push_sq, PieceType::kRook, PieceType::kPawn};
          *out++ = {sq, push_sq, PieceType::kQueen, PieceType::kPawn};
        } else {
          *out++ = {sq, push_sq, PieceType::kNone, PieceType::kPawn};
        }
      }
      if (r == start_rank) {
        int dp_sq = sq + (2 * push_dir);
        Bitboard dp_bb = square_bb(dp_sq);
        if (!(occ & dp_bb) && (dp_bb & target_mask) &&
            (!pinned || (dp_bb & pin_ray))) {
          *out++ = {sq, dp_sq, PieceType::kNone, PieceType::kPawn};
        }
      }
    }

    Bitboard cap_targets =
        kAttacks.pawn_attacks[!is_white][sq] & them_bb & target_mask;
    if (pinned) cap_targets &= pin_ray;
    while (cap_targets) {
      int cap_sq = std::countr_zero(cap_targets);
      cap_targets &= cap_targets - 1;
      if (r == promo_rank) {
        *out++ = {sq, cap_sq, PieceType::kKnight, PieceType::kPawn};
        *out++ = {sq, cap_sq, PieceType::kBishop, PieceType::kPawn};
        *out++ = {sq, cap_sq, PieceType::kRook, PieceType::kPawn};
        *out++ = {sq, cap_sq, PieceType::kQueen, PieceType::kPawn};
      } else {
        *out++ = {sq, cap_sq, PieceType::kNone, PieceType::kPawn};
      }
    }
  }
}

void ChessState::add_en_passant_fast(ChessMove*& out, int king_sq,
                                     Bitboard occ) const {
  if (en_passant_square_ == -1) return;

  Bitboard pawns = pieces_[static_cast<size_t>(PieceType::kPawn)] & us();
  bool is_white = side_to_move_ == Color::kWhite;
  int ep_sq = en_passant_square_;
  Color them = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));

  Bitboard from_sqs_bb =
      kAttacks.pawn_attacks[static_cast<size_t>(them)][ep_sq] & pawns;

  int ep_captured = ep_sq + (is_white ? -8 : 8);
  Bitboard ep_captured_bb = square_bb(ep_captured);
  Bitboard ep_to_bb = square_bb(ep_sq);

  while (from_sqs_bb) {
    int from = std::countr_zero(from_sqs_bb);
    from_sqs_bb &= from_sqs_bb - 1;
    Bitboard from_bb = square_bb(from);
    Bitboard slider_occ = (occ & ~from_bb & ~ep_captured_bb) | ep_to_bb;
    if (!is_attacked(king_sq, them, slider_occ, ep_captured_bb))
      *out++ = {from, ep_sq, PieceType::kNone, PieceType::kPawn};
  }
}

void ChessState::add_knight_moves_fast(ChessMove*& out, Bitboard pinned_bb,
                                       const Bitboard* pin_rays,
                                       Bitboard target_mask) const {
  add_piece_moves_fast(
      out, pieces_[static_cast<size_t>(PieceType::kKnight)] & us(),
      [this](int sq) { return kAttacks.knight[sq]; }, PieceType::kKnight,
      pinned_bb, pin_rays, target_mask);
}

void ChessState::add_king_moves_fast(ChessMove*& out, int king_sq,
                                     Bitboard occ) const {
  Color them = static_cast<Color>(1 - static_cast<size_t>(side_to_move_));
  Bitboard occ_without = occ & ~square_bb(king_sq);
  Bitboard targets = kAttacks.king[king_sq] & ~us();
  while (targets) {
    int t = std::countr_zero(targets);
    targets &= targets - 1;
    if (!is_attacked(t, them, occ_without))
      *out++ = {king_sq, t, PieceType::kNone, PieceType::kKing};
  }

  if (side_to_move_ == Color::kWhite) {
    if ((castling_rights_ & 1) && !(occ & 0x60) &&
        !is_attacked(4, Color::kBlack) && !is_attacked(5, Color::kBlack) &&
        !is_attacked(6, Color::kBlack))
      *out++ = {square(File::E, Rank::R1), square(File::G, Rank::R1),
                PieceType::kNone, PieceType::kKing};
    if ((castling_rights_ & 2) && !(occ & 0x0E) &&
        !is_attacked(4, Color::kBlack) && !is_attacked(3, Color::kBlack) &&
        !is_attacked(2, Color::kBlack))
      *out++ = {square(File::E, Rank::R1), square(File::C, Rank::R1),
                PieceType::kNone, PieceType::kKing};
  } else {
    if ((castling_rights_ & 4) && !(occ & 0x6000000000000000) &&
        !is_attacked(60, Color::kWhite) && !is_attacked(61, Color::kWhite) &&
        !is_attacked(62, Color::kWhite))
      *out++ = {square(File::E, Rank::R8), square(File::G, Rank::R8),
                PieceType::kNone, PieceType::kKing};
    if ((castling_rights_ & 8) && !(occ & 0x0E00000000000000) &&
        !is_attacked(60, Color::kWhite) && !is_attacked(59, Color::kWhite) &&
        !is_attacked(58, Color::kWhite))
      *out++ = {square(File::E, Rank::R8), square(File::C, Rank::R8),
                PieceType::kNone, PieceType::kKing};
  }
}

void ChessState::add_bishop_moves_fast(ChessMove*& out, Bitboard occ,
                                       Bitboard pinned_bb,
                                       const Bitboard* pin_rays,
                                       Bitboard target_mask) const {
  add_piece_moves_fast(
      out, pieces_[static_cast<size_t>(PieceType::kBishop)] & us(),
      [this, occ](int sq) { return bishop_targets(sq, occ); },
      PieceType::kBishop, pinned_bb, pin_rays, target_mask);
}

void ChessState::add_rook_moves_fast(ChessMove*& out, Bitboard occ,
                                     Bitboard pinned_bb,
                                     const Bitboard* pin_rays,
                                     Bitboard target_mask) const {
  add_piece_moves_fast(
      out, pieces_[static_cast<size_t>(PieceType::kRook)] & us(),
      [this, occ](int sq) { return rook_targets(sq, occ); }, PieceType::kRook,
      pinned_bb, pin_rays, target_mask);
}

void ChessState::add_queen_moves_fast(ChessMove*& out, Bitboard occ,
                                      Bitboard pinned_bb,
                                      const Bitboard* pin_rays,
                                      Bitboard target_mask) const {
  add_piece_moves_fast(
      out, pieces_[static_cast<size_t>(PieceType::kQueen)] & us(),
      [this, occ](int sq) { return queen_targets(sq, occ); }, PieceType::kQueen,
      pinned_bb, pin_rays, target_mask);
}

int ChessState::legal_moves_fast(ChessMove* result) const {
  Color us = side_to_move_;
  Color them = static_cast<Color>(1 - static_cast<size_t>(us));
  int king_sq =
      std::countr_zero(pieces_[static_cast<size_t>(PieceType::kKing)] &
                       colors_[static_cast<size_t>(us)]);
  Bitboard occ = occupied();

  Bitboard checkers = compute_checkers(king_sq, them);
  int num_checks = std::popcount(checkers);

  ChessMove* out = result;

  if (num_checks >= 2) {
    add_king_moves_fast(out, king_sq, occ);
    return static_cast<int>(out - result);
  }

  Bitboard pin_rays[64];
  Bitboard pinned = compute_pinned(king_sq, us, them, pin_rays);

  if (num_checks == 1) {
    int checker_sq = std::countr_zero(checkers);
    Bitboard check_ray =
        between_bb(king_sq, checker_sq) | square_bb(checker_sq);

    add_king_moves_fast(out, king_sq, occ);
    add_pawn_moves_fast(out, pinned, pin_rays, check_ray);
    add_knight_moves_fast(out, pinned, pin_rays, check_ray);
    add_bishop_moves_fast(out, occ, pinned, pin_rays, check_ray);
    add_rook_moves_fast(out, occ, pinned, pin_rays, check_ray);
    add_queen_moves_fast(out, occ, pinned, pin_rays, check_ray);
    add_en_passant_fast(out, king_sq, occ);
  } else {
    Bitboard all = ~Bitboard{0};
    add_pawn_moves_fast(out, pinned, pin_rays, all);
    add_knight_moves_fast(out, pinned, pin_rays, all);
    add_bishop_moves_fast(out, occ, pinned, pin_rays, all);
    add_rook_moves_fast(out, occ, pinned, pin_rays, all);
    add_queen_moves_fast(out, occ, pinned, pin_rays, all);
    add_king_moves_fast(out, king_sq, occ);
    add_en_passant_fast(out, king_sq, occ);
  }

  return static_cast<int>(out - result);
}

ChessState::MoveUndo ChessState::make_move_perft(ChessMove m) {
  int us = static_cast<int>(side_to_move_);
  int them = 1 - us;
  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  MoveUndo undo;
  undo.old_castling_rights = castling_rights_;
  undo.old_en_passant_square = en_passant_square_;
  undo.captured_piece = PieceType::kNone;

  PieceType pt = m.piece;

  Bitboard captured_occ = to_bb & colors_[them];
  if (captured_occ) {
    for (int i = 1; i <= 6; ++i) {
      if (pieces_[i] & captured_occ) {
        undo.captured_piece = static_cast<PieceType>(i);
        pieces_[i] ^= to_bb;
        break;
      }
    }
    colors_[them] ^= to_bb;
  }

  if (pt == PieceType::kPawn && m.to == en_passant_square_) {
    int ep_pawn_sq = en_passant_square_ + (us == 0 ? -8 : 8);
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
    colors_[them] ^= square_bb(ep_pawn_sq);
    undo.captured_piece = PieceType::kPawn;
  }

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[us] ^= from_bb | to_bb;

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

  side_to_move_ = static_cast<Color>(them);

  return undo;
}

void ChessState::unmake_move_perft(ChessMove m, const MoveUndo& undo) {
  int us = 1 - static_cast<int>(side_to_move_);
  int them = 1 - us;
  side_to_move_ = static_cast<Color>(us);

  Bitboard from_bb = square_bb(m.from);
  Bitboard to_bb = square_bb(m.to);

  PieceType pt = m.piece;
  if (m.promotion != PieceType::kNone) pt = PieceType::kPawn;

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[us] ^= from_bb | to_bb;

  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
  }

  if (undo.captured_piece != PieceType::kNone) {
    if (pt == PieceType::kPawn && m.to == undo.old_en_passant_square) {
      int ep_pawn_sq = undo.old_en_passant_square + (us == 0 ? -8 : 8);
      pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
      colors_[them] ^= square_bb(ep_pawn_sq);
    } else {
      pieces_[static_cast<size_t>(undo.captured_piece)] ^= to_bb;
      colors_[them] ^= to_bb;
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
}

size_t ChessState::PerftFast(size_t depth) {
  if (depth == 0) return 1;
  ChessMove moves[ChessMove::kMaxMoves];
  int n = legal_moves_fast(moves);
  if (depth == 1) return static_cast<size_t>(n);
  size_t nodes = 0;
  for (int i = 0; i < n; ++i) {
    MoveUndo undo = make_move_perft(moves[i]);
    nodes += PerftFast(depth - 1);
    unmake_move_perft(moves[i], undo);
  }
  return nodes;
}
