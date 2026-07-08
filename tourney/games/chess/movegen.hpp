#include <immintrin.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

enum class PieceType : uint8_t {
  kNone,
  kPawn,
  kKnight,
  kBishop,
  kRook,
  kQueen,
  kKing
};

struct ChessMove {
  static constexpr int kMaxMoves = 218;

  int from;
  int to;
  PieceType promotion;
  PieceType piece;
  PieceType captured;

  ChessMove() = default;
  ChessMove(int from, int to, PieceType promotion = PieceType::kNone,
            PieceType piece = PieceType::kNone,
            PieceType captured = PieceType::kNone)
      : from(from),
        to(to),
        promotion(promotion),
        piece(piece),
        captured(captured) {}
};

class ChessState final {
 public:
  using MoveT = ChessMove;

  struct MoveUndo {
    PieceType captured_piece;
    uint8_t old_castling_rights;
    int old_en_passant_square;
    int old_halfmove_clock;
  };

  void MakeMove(const ChessMove& move, MoveUndo& undo);
  void UnmakeMove(const ChessMove& move, const MoveUndo& undo);

  static ChessState initial_position();
  static ChessState from_fen(std::string_view fen);

  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] std::optional<ChessMove> Parse(const std::string& input) const;
  [[nodiscard]] size_t FillLegalMoves(ChessMove* buffer, size_t capacity) const;
  [[nodiscard]] bool IsOver() const;

 private:
  enum class Color : uint8_t { kWhite, kBlack };
  enum class File : uint8_t { A, B, C, D, E, F, G, H };
  enum class Rank : uint8_t { R1, R2, R3, R4, R5, R6, R7, R8 };

  using Bitboard = uint64_t;

  static constexpr Bitboard square_bb(int sq) { return Bitboard{1} << sq; }
  static constexpr int square(File f, Rank r) {
    return static_cast<int>(f) + (8 * static_cast<int>(r));
  }

  struct AttackTables {
    std::array<Bitboard, 64> knight{};
    std::array<Bitboard, 64> king{};
    std::array<Bitboard, 64> pawn_attacks[2]{};
    std::array<std::array<Bitboard, 64>, 64> between{};
  };

  static AttackTables compute_attack_tables();
  inline static const AttackTables kAttacks = compute_attack_tables();

  [[gnu::always_inline]] static Bitboard rook_targets(int sq, Bitboard occ) {
    Bitboard idx = _pext_u64(occ, rook_masks_[sq]);
    return rook_table_[rook_offset_[sq] + idx];
  }
  [[gnu::always_inline]] static Bitboard bishop_targets(int sq, Bitboard occ) {
    Bitboard idx = _pext_u64(occ, bishop_masks_[sq]);
    return bishop_table_[bishop_offset_[sq] + idx];
  }
  [[gnu::always_inline]] static Bitboard queen_targets(int sq, Bitboard occ) {
    return rook_targets(sq, occ) | bishop_targets(sq, occ);
  }

  [[gnu::always_inline]] static Bitboard between_bb(int sq1, int sq2) {
    return kAttacks.between[sq1][sq2];
  }

  static void init_magic_tables();
  static void init_magic_table(bool bishop);
  static Bitboard rook_masks_[64];
  static Bitboard* rook_table_;
  static int rook_offset_[64];
  static Bitboard bishop_masks_[64];
  static Bitboard* bishop_table_;
  static int bishop_offset_[64];
  static bool magic_initialized_;

  template <Color kBy>
  [[gnu::always_inline]] [[nodiscard]] inline bool is_attacked(
      int sq, Bitboard slider_occ, Bitboard ignore = 0) const {
    constexpr auto kByI = static_cast<size_t>(kBy);
    constexpr size_t kOther = 1 - kByI;

    Bitboard enemy = colors_[kByI] & ~ignore;

    if (kAttacks.pawn_attacks[kOther][sq] &
        pieces_[static_cast<size_t>(PieceType::kPawn)] & enemy) {
      return true;
    }
    if (kAttacks.knight[sq] & pieces_[static_cast<size_t>(PieceType::kKnight)] &
        enemy) {
      return true;
    }
    if (kAttacks.king[sq] & pieces_[static_cast<size_t>(PieceType::kKing)] &
        enemy) {
      return true;
    }
    if (bishop_targets(sq, slider_occ) &
        (pieces_[static_cast<size_t>(PieceType::kBishop)] |
         pieces_[static_cast<size_t>(PieceType::kQueen)]) &
        enemy) {
      return true;
    }
    if (rook_targets(sq, slider_occ) &
        (pieces_[static_cast<size_t>(PieceType::kRook)] |
         pieces_[static_cast<size_t>(PieceType::kQueen)]) &
        enemy) {
      return true;
    }
    return false;
  }

  [[nodiscard]] bool is_attacked(int sq, Color by,
                                 Bitboard slider_occ = ~Bitboard{0},
                                 Bitboard ignore = 0) const {
    if (slider_occ == ~Bitboard{0}) {
      slider_occ = occupied();
    }
    if (by == Color::kWhite) {
      return is_attacked<Color::kWhite>(sq, slider_occ, ignore);
    }
    return is_attacked<Color::kBlack>(sq, slider_occ, ignore);
  }

  template <Color kThem>
  [[gnu::always_inline]] [[nodiscard]] inline Bitboard compute_checkers(
      int king_sq) const {
    constexpr auto kThemI = static_cast<size_t>(kThem);
    constexpr size_t kUs = 1 - kThemI;
    Bitboard them_bb = colors_[kThemI];
    Bitboard occ = occupied();

    Bitboard knight_checkers =
        kAttacks.knight[king_sq] &
        pieces_[static_cast<size_t>(PieceType::kKnight)] & them_bb;

    Bitboard pawn_checkers = kAttacks.pawn_attacks[kUs][king_sq] &
                             pieces_[static_cast<size_t>(PieceType::kPawn)] &
                             them_bb;

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

  template <Color kUs, Color kThem>
  [[gnu::always_inline]] inline Bitboard compute_pinned(
      int king_sq, Bitboard* pin_rays) const {
    constexpr auto kUsI = static_cast<size_t>(kUs);
    constexpr auto kThemI = static_cast<size_t>(kThem);
    Bitboard us_bb = colors_[kUsI];
    Bitboard them_bb = colors_[kThemI];

    Bitboard enemy_rooks = (pieces_[static_cast<size_t>(PieceType::kRook)] |
                            pieces_[static_cast<size_t>(PieceType::kQueen)]) &
                           them_bb;
    Bitboard enemy_bishops = (pieces_[static_cast<size_t>(PieceType::kBishop)] |
                              pieces_[static_cast<size_t>(PieceType::kQueen)]) &
                             them_bb;

    if (!(rook_targets(king_sq, 0) & enemy_rooks) &&
        !(bishop_targets(king_sq, 0) & enemy_bishops)) {
      return 0;
    }

    const Bitboard rook_pinners = rook_targets(king_sq, them_bb) & enemy_rooks;
    const Bitboard bishop_pinners =
        bishop_targets(king_sq, them_bb) & enemy_bishops;

    Bitboard pinners = rook_pinners | bishop_pinners;
    Bitboard pinned = 0;

    while (pinners) {
      const int psq = std::countr_zero(pinners);
      pinners &= pinners - 1;
      const Bitboard between = between_bb(king_sq, psq);
      const Bitboard friends = between & us_bb;
      if (friends && !(friends & (friends - 1))) {
        const int fsq = std::countr_zero(friends);
        pinned |= square_bb(fsq);
        if (pin_rays) {
          pin_rays[fsq] = between | square_bb(psq);
        }
      }
    }
    return pinned;
  }

  template <typename F>
  [[gnu::always_inline]] inline void add_piece_moves_fast(
      ChessMove*& out, Bitboard pieces_bb, F&& get_targets, PieceType pt,
      Bitboard pinned_bb = 0, const Bitboard* pin_rays = nullptr,
      Bitboard target_mask = ~Bitboard{0}) const {
    Bitboard unpinned = pieces_bb & ~pinned_bb;
    while (unpinned) {
      const int sq = std::countr_zero(unpinned);
      unpinned &= unpinned - 1;
      Bitboard targets = get_targets(sq) & ~us() & target_mask;
      while (targets) {
        const int t = std::countr_zero(targets);
        targets &= targets - 1;
        *out++ = ChessMove(sq, t, PieceType::kNone, pt);
      }
    }
    if (!pin_rays) {
      return;
    }
    Bitboard pinned = pieces_bb & pinned_bb;
    while (pinned) {
      const int sq = std::countr_zero(pinned);
      pinned &= pinned - 1;
      Bitboard targets = get_targets(sq) & ~us() & target_mask & pin_rays[sq];
      while (targets) {
        const int t = std::countr_zero(targets);
        targets &= targets - 1;
        *out++ = ChessMove(sq, t, PieceType::kNone, pt);
      }
    }
  }

  [[gnu::always_inline]] static bool pin_allows(bool pinned, Bitboard pin_ray,
                                                Bitboard to_bb) {
    return !pinned || (to_bb & pin_ray) != 0;
  }

  [[nodiscard]] Bitboard occupied() const { return colors_[0] | colors_[1]; }
  [[nodiscard]] Bitboard us() const {
    return colors_[static_cast<size_t>(side_to_move_)];
  }
  [[nodiscard]] Bitboard them() const {
    return colors_[1 - static_cast<size_t>(side_to_move_)];
  }

  template <bool kWhite>
  [[gnu::always_inline]] void add_pawn_moves_fast_impl(
      ChessMove*& out, Bitboard pinned_bb, const Bitboard* pin_rays,
      Bitboard target_mask) const {
    Bitboard pawns = pieces_[static_cast<size_t>(PieceType::kPawn)] & us();
    const Bitboard occ = occupied();
    const Bitboard them_bb = them();
    constexpr int kPushDir = kWhite ? 8 : -8;
    constexpr int kStartRank = kWhite ? 1 : 6;
    constexpr int kPromoRank = kWhite ? 6 : 1;
    constexpr int kAttackIdx = kWhite ? 0 : 1;

    while (pawns) {
      const int sq = std::countr_zero(pawns);
      pawns &= pawns - 1;
      const int r = sq / 8;
      const bool pinned = (pinned_bb & square_bb(sq)) != 0;
      const Bitboard pin_ray = pinned && pin_rays ? pin_rays[sq] : ~Bitboard{0};

      int push_sq = sq + kPushDir;
      Bitboard push_bb = square_bb(push_sq);
      if (!(occ & push_bb)) {
        if ((push_bb & target_mask) && pin_allows(pinned, pin_ray, push_bb)) {
          if (r == kPromoRank) {
            *out++ = {sq, push_sq, PieceType::kKnight, PieceType::kPawn};
            *out++ = {sq, push_sq, PieceType::kBishop, PieceType::kPawn};
            *out++ = {sq, push_sq, PieceType::kRook, PieceType::kPawn};
            *out++ = {sq, push_sq, PieceType::kQueen, PieceType::kPawn};
          } else {
            *out++ = {sq, push_sq, PieceType::kNone, PieceType::kPawn};
          }
        }
        if (r == kStartRank) {
          int dp_sq = sq + (2 * kPushDir);
          Bitboard dp_bb = square_bb(dp_sq);
          if (!(occ & dp_bb) && (dp_bb & target_mask) &&
              pin_allows(pinned, pin_ray, dp_bb)) {
            *out++ = {sq, dp_sq, PieceType::kNone, PieceType::kPawn};
          }
        }
      }

      Bitboard cap_targets =
          kAttacks.pawn_attacks[kAttackIdx][sq] & them_bb & target_mask;
      if (pinned) {
        cap_targets &= pin_ray;
      }
      while (cap_targets) {
        const int cap_sq = std::countr_zero(cap_targets);
        cap_targets &= cap_targets - 1;
        if (r == kPromoRank) {
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

  template <bool kWhite>
  [[gnu::always_inline]] void add_en_passant_fast_impl(ChessMove*& out,
                                                       int king_sq,
                                                       Bitboard occ) const {
    if (en_passant_square_ == -1) {
      return;
    }

    constexpr Color kThem = kWhite ? Color::kBlack : Color::kWhite;
    Bitboard pawns = pieces_[static_cast<size_t>(PieceType::kPawn)] & us();
    const int ep_sq = en_passant_square_;

    Bitboard from_sqs_bb =
        kAttacks.pawn_attacks[static_cast<size_t>(kThem)][ep_sq] & pawns;

    constexpr int kCapturedOffset = kWhite ? -8 : 8;
    int ep_captured = ep_sq + kCapturedOffset;
    Bitboard ep_captured_bb = square_bb(ep_captured);
    const Bitboard ep_to_bb = square_bb(ep_sq);

    while (from_sqs_bb) {
      const int from = std::countr_zero(from_sqs_bb);
      from_sqs_bb &= from_sqs_bb - 1;
      const Bitboard from_bb = square_bb(from);
      Bitboard slider_occ = (occ & ~from_bb & ~ep_captured_bb) | ep_to_bb;
      if (!is_attacked<kThem>(king_sq, slider_occ, ep_captured_bb)) {
        *out++ = {from, ep_sq, PieceType::kNone, PieceType::kPawn};
      }
    }
  }

  [[gnu::always_inline]] void add_knight_moves_fast_impl(
      ChessMove*& out, Bitboard pinned_bb, const Bitboard* pin_rays,
      Bitboard target_mask) const {
    add_piece_moves_fast(
        out, pieces_[static_cast<size_t>(PieceType::kKnight)] & us(),
        [](int sq) { return kAttacks.knight[sq]; }, PieceType::kKnight,
        pinned_bb, pin_rays, target_mask);
  }

  template <bool kWhite>
  [[gnu::always_inline]] void add_king_moves_fast_impl(ChessMove*& out,
                                                       int king_sq,
                                                       Bitboard occ) const {
    constexpr Color kThem = kWhite ? Color::kBlack : Color::kWhite;
    Bitboard occ_without = occ & ~square_bb(king_sq);
    Bitboard targets = kAttacks.king[king_sq] & ~us();
    while (targets) {
      int t = std::countr_zero(targets);
      targets &= targets - 1;
      if (!is_attacked<kThem>(t, occ_without))
        *out++ = {king_sq, t, PieceType::kNone, PieceType::kKing};
    }

    if constexpr (kWhite) {
      if ((castling_rights_ & 1) && !(occ & 0x60) &&
          !is_attacked<Color::kBlack>(4, occ) &&
          !is_attacked<Color::kBlack>(5, occ) &&
          !is_attacked<Color::kBlack>(6, occ))
        *out++ = {square(File::E, Rank::R1), square(File::G, Rank::R1),
                  PieceType::kNone, PieceType::kKing};
      if ((castling_rights_ & 2) && !(occ & 0x0E) &&
          !is_attacked<Color::kBlack>(4, occ) &&
          !is_attacked<Color::kBlack>(3, occ) &&
          !is_attacked<Color::kBlack>(2, occ))
        *out++ = {square(File::E, Rank::R1), square(File::C, Rank::R1),
                  PieceType::kNone, PieceType::kKing};
    } else {
      if ((castling_rights_ & 4) && !(occ & 0x6000000000000000) &&
          !is_attacked<Color::kWhite>(60, occ) &&
          !is_attacked<Color::kWhite>(61, occ) &&
          !is_attacked<Color::kWhite>(62, occ))
        *out++ = {square(File::E, Rank::R8), square(File::G, Rank::R8),
                  PieceType::kNone, PieceType::kKing};
      if ((castling_rights_ & 8) && !(occ & 0x0E00000000000000) &&
          !is_attacked<Color::kWhite>(60, occ) &&
          !is_attacked<Color::kWhite>(59, occ) &&
          !is_attacked<Color::kWhite>(58, occ))
        *out++ = {square(File::E, Rank::R8), square(File::C, Rank::R8),
                  PieceType::kNone, PieceType::kKing};
    }
  }

  [[gnu::always_inline]] void add_bishop_moves_fast_impl(
      ChessMove*& out, Bitboard occ, Bitboard pinned_bb,
      const Bitboard* pin_rays, Bitboard target_mask) const {
    add_piece_moves_fast(
        out, pieces_[static_cast<size_t>(PieceType::kBishop)] & us(),
        [occ](int sq) { return bishop_targets(sq, occ); }, PieceType::kBishop,
        pinned_bb, pin_rays, target_mask);
  }

  [[gnu::always_inline]] void add_rook_moves_fast_impl(
      ChessMove*& out, Bitboard occ, Bitboard pinned_bb,
      const Bitboard* pin_rays, Bitboard target_mask) const {
    add_piece_moves_fast(
        out, pieces_[static_cast<size_t>(PieceType::kRook)] & us(),
        [occ](int sq) { return rook_targets(sq, occ); }, PieceType::kRook,
        pinned_bb, pin_rays, target_mask);
  }

  [[gnu::always_inline]] void add_queen_moves_fast_impl(
      ChessMove*& out, Bitboard occ, Bitboard pinned_bb,
      const Bitboard* pin_rays, Bitboard target_mask) const {
    add_piece_moves_fast(
        out, pieces_[static_cast<size_t>(PieceType::kQueen)] & us(),
        [occ](int sq) { return queen_targets(sq, occ); }, PieceType::kQueen,
        pinned_bb, pin_rays, target_mask);
  }

  template <bool kWhite>
  [[gnu::always_inline]] size_t FillLegalMovesImpl(ChessMove* buffer) const {
    ChessMove* result = buffer;
    constexpr Color kUs = kWhite ? Color::kWhite : Color::kBlack;
    constexpr Color kThem = kWhite ? Color::kBlack : Color::kWhite;
    int king_sq =
        std::countr_zero(pieces_[static_cast<size_t>(PieceType::kKing)] &
                         colors_[static_cast<size_t>(kUs)]);
    Bitboard occ = occupied();

    Bitboard checkers = compute_checkers<kThem>(king_sq);
    const int num_checks = std::popcount(checkers);

    ChessMove* out = result;

    if (num_checks >= 2) {
      add_king_moves_fast_impl<kWhite>(out, king_sq, occ);
      return static_cast<size_t>(out - result);
    }

    Bitboard pin_rays[64];
    Bitboard pinned = compute_pinned<kUs, kThem>(king_sq, pin_rays);

    if (num_checks == 1) {
      const int checker_sq = std::countr_zero(checkers);
      Bitboard check_ray =
          between_bb(king_sq, checker_sq) | square_bb(checker_sq);

      add_king_moves_fast_impl<kWhite>(out, king_sq, occ);
      add_pawn_moves_fast_impl<kWhite>(out, pinned, pin_rays, check_ray);
      add_knight_moves_fast_impl(out, pinned, pin_rays, check_ray);
      add_bishop_moves_fast_impl(out, occ, pinned, pin_rays, check_ray);
      add_rook_moves_fast_impl(out, occ, pinned, pin_rays, check_ray);
      add_queen_moves_fast_impl(out, occ, pinned, pin_rays, check_ray);
      add_en_passant_fast_impl<kWhite>(out, king_sq, occ);
    } else {
      Bitboard all = ~Bitboard{0};
      add_pawn_moves_fast_impl<kWhite>(out, pinned, pin_rays, all);
      add_knight_moves_fast_impl(out, pinned, pin_rays, all);
      add_bishop_moves_fast_impl(out, occ, pinned, pin_rays, all);
      add_rook_moves_fast_impl(out, occ, pinned, pin_rays, all);
      add_queen_moves_fast_impl(out, occ, pinned, pin_rays, all);
      add_king_moves_fast_impl<kWhite>(out, king_sq, occ);
      add_en_passant_fast_impl<kWhite>(out, king_sq, occ);
    }

    return static_cast<size_t>(out - result);
  }

  [[nodiscard]] PieceType piece_type_at(int sq) const;

  std::array<Bitboard, 7> pieces_{};
  std::array<Bitboard, 2> colors_{};
  Color side_to_move_ = Color::kWhite;
  uint8_t castling_rights_ = 0xF;
  int en_passant_square_ = -1;
  int halfmove_clock_ = 0;
  int fullmove_number_ = 1;
};

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
    const int df = (s % 8) - ((s - dir) % 8);
    if (df > 1 || df < -1) {
      break;
    }
    targets |= (BB{1} << s);
    if (occ & (BB{1} << s)) {
      break;
    }
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
      if ((s % 8) - (prev % 8) > 1 || (prev % 8) - (s % 8) > 1) {
        break;
      }
      int next = s + dir;
      bool terminal = !(0 <= next && next < 64) || ((next % 8) - (s % 8) > 1) ||
                      ((s % 8) - (next % 8) > 1);
      if (terminal) {
        break;
      }
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
    if (idx & 1) {
      result |= (BB{1} << bit);
    }
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
  for (int di = 0; di < 4; ++di) {
    attacks |= sliding_targets(occ, sq, dirs[di]);
  }
  return attacks;
}

// Castling rights bits (same encoding as ChessState::castling_rights_):
//   1 = WK, 2 = WQ, 4 = BK, 8 = BQ
// castle_clear[sq] = mask of rights to revoke when a piece moves from/to `sq`.
//   - From a1 (0) or to a1: revoke WQ (2).
//   - From h1 (7) or to h1: revoke WK (1).
//   - From a8 (56) or to a8: revoke BQ (8).
//   - From h8 (63) or to h8: revoke BK (4).
//   - From e1 (4): revoke both WK and WQ (3).
//   - From e8 (60): revoke both BK and BQ (12).
constexpr std::array<uint8_t, 64> castle_clear = []() {
  std::array<uint8_t, 64> t{};
  t[0] = 2;
  t[7] = 1;
  t[56] = 8;
  t[63] = 4;
  t[4] = 3;
  t[60] = 12;
  return t;
}();

// Castling rook XOR: for a king move (from_e1_to_g1 etc.), the (rook_from,
// rook_to) squares whose bits must be XOR'd into pieces_[kRook]/colors_[].
// castle_rook[castle_idx] = {from, to, color_idx}; castle_idx matches the
// castling_rights bit that the castle move belongs to (1=WK, 2=WQ, 4=BK, 8=BQ).
struct CastleRookMove {
  int rook_from;
  int rook_to;
  int color_idx;
};

// Indexed by the single set bit of castling_rights_ that triggers the castle.
constexpr std::array<CastleRookMove, 16> castle_rook = []() {
  std::array<CastleRookMove, 16> t{};
  t[1] = {.rook_from = 7, .rook_to = 5, .color_idx = 0};
  t[2] = {.rook_from = 0, .rook_to = 3, .color_idx = 0};
  t[4] = {.rook_from = 63, .rook_to = 61, .color_idx = 1};
  t[8] = {.rook_from = 56, .rook_to = 59, .color_idx = 1};
  return t;
}();

// Maps a (king_from, king_to) pair to the castling-rights bit (1, 2, 4, or 8),
// or 0 if the king move is not a castle. Index = (from << 6) | to; squashed
// into a 4096-entry table for direct lookup. Only 4 of the entries are nonzero.
constexpr std::array<uint8_t, 4096> castle_index = []() {
  std::array<uint8_t, 4096> t{};
  // WK: e1 (4) -> g1 (6). bit 1.
  t[(4 << 6) | 6] = 1;
  // WQ: e1 (4) -> c1 (2). bit 2.
  t[(4 << 6) | 2] = 2;
  // BK: e8 (60) -> g8 (62). bit 4.
  t[(60 << 6) | 62] = 4;
  // BQ: e8 (60) -> c8 (58). bit 8.
  t[(60 << 6) | 58] = 8;
  return t;
}();

}  // anonymous namespace

void ChessState::init_magic_table(bool bishop) {
  int total = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard mask = relevant_mask(sq, bishop);
    const int bits = std::popcount(mask);
    const int n = 1 << bits;
    if (bishop) {
      bishop_offset_[sq] = total;
    } else {
      rook_offset_[sq] = total;
    }
    total += n;
    if (bishop) {
      bishop_masks_[sq] = mask;
    } else {
      rook_masks_[sq] = mask;
    }
  }
  auto* table = new Bitboard[total];
  if (bishop) {
    bishop_table_ = table;
  } else {
    rook_table_ = table;
  }
  for (int sq = 0; sq < 64; ++sq) {
    const Bitboard mask = bishop ? bishop_masks_[sq] : rook_masks_[sq];
    const int bits = std::popcount(mask);
    const int n = 1 << bits;
    const int offset = bishop ? bishop_offset_[sq] : rook_offset_[sq];
    for (int i = 0; i < n; ++i) {
      const Bitboard occ = expand(i, mask);
      table[offset + i] = attacks_for_occ(occ, sq, bishop);
    }
  }
}

void ChessState::init_magic_tables() {
  if (magic_initialized_) {
    return;
  }
  magic_initialized_ = true;
  init_magic_table(false);
  init_magic_table(true);
}

ChessState::AttackTables ChessState::compute_attack_tables() {
  init_magic_tables();
  AttackTables a{};
  constexpr int kd[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                            {1, -2},  {1, 2},  {2, -1},  {2, 1}};
  constexpr int kKingDeltas[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                     {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8;
    const int f = sq % 8;
    for (const auto& d : kd) {
      const int nr = r + d[0];
      const int nf = f + d[1];
      if (0 <= nr && nr < 8 && 0 <= nf && nf < 8) {
        a.knight[sq] |= square_bb((nr * 8) + nf);
      }
    }
    for (const auto& d : kKingDeltas) {
      const int nr = r + d[0];
      const int nf = f + d[1];
      if (0 <= nr && nr < 8 && 0 <= nf && nf < 8) {
        a.king[sq] |= square_bb((nr * 8) + nf);
      }
    }

    if (f > 0 && r < 7) {
      a.pawn_attacks[0][sq] |= square_bb(sq + 7);
    }
    if (f < 7 && r < 7) {
      a.pawn_attacks[0][sq] |= square_bb(sq + 9);
    }
    if (f < 7 && r > 0) {
      a.pawn_attacks[1][sq] |= square_bb(sq - 7);
    }
    if (f > 0 && r > 0) {
      a.pawn_attacks[1][sq] |= square_bb(sq - 9);
    }
  }

  // Depends on rook_table_/bishop_table_ populated by init_magic_tables()
  // above.
  for (int sq1 = 0; sq1 < 64; ++sq1) {
    const int f1 = sq1 & 7, r1 = sq1 >> 3;
    for (int sq2 = 0; sq2 < 64; ++sq2) {
      const int f2 = sq2 & 7, r2 = sq2 >> 3;
      if (f1 == f2 || r1 == r2) {
        const Bitboard occ = square_bb(sq1) | square_bb(sq2);
        a.between[sq1][sq2] = rook_targets(sq1, occ) & rook_targets(sq2, occ);
      } else {
        const int df = f1 > f2 ? f1 - f2 : f2 - f1;
        const int dr = r1 > r2 ? r1 - r2 : r2 - r1;
        if (df == dr) {
          const Bitboard occ = square_bb(sq1) | square_bb(sq2);
          a.between[sq1][sq2] =
              bishop_targets(sq1, occ) & bishop_targets(sq2, occ);
        }
      }
    }
  }

  return a;
}

void ChessState::MakeMove(const ChessMove& m, MoveUndo& undo) {
  const int us = static_cast<int>(side_to_move_);
  const int them = 1 - us;
  const bool is_white = us == 0;
  const Bitboard from_bb = square_bb(m.from);
  const Bitboard to_bb = square_bb(m.to);

  undo = MoveUndo{};
  undo.old_castling_rights = castling_rights_;
  undo.old_en_passant_square = en_passant_square_;
  undo.old_halfmove_clock = halfmove_clock_;
  undo.captured_piece = PieceType::kNone;

  PieceType pt = m.piece;

  Bitboard captured_occ = to_bb & colors_[them];
  if (captured_occ != 0) {
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
    const int ep_pawn_sq = en_passant_square_ + (is_white ? -8 : 8);
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
    const uint8_t idx = castle_index[(m.from << 6) | m.to];
    if (idx != 0) {
      const auto& cr = castle_rook[idx];
      const Bitboard rook = square_bb(cr.rook_from);
      const Bitboard rook_to = square_bb(cr.rook_to);
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[cr.color_idx] ^= rook | rook_to;
    }
  }

  castling_rights_ &= ~(castle_clear[m.from] | castle_clear[m.to]);

  if (pt == PieceType::kPawn && (m.to - m.from == 16 || m.to - m.from == -16)) {
    en_passant_square_ = (m.from + m.to) / 2;
  } else {
    en_passant_square_ = -1;
  }

  side_to_move_ = static_cast<Color>(them);
}

void ChessState::UnmakeMove(const ChessMove& m, const MoveUndo& undo) {
  const int us = 1 - static_cast<int>(side_to_move_);
  const int them = 1 - us;
  const bool is_white = us == 0;
  side_to_move_ = static_cast<Color>(us);

  const Bitboard from_bb = square_bb(m.from);
  const Bitboard to_bb = square_bb(m.to);

  PieceType pt = m.piece;
  if (m.promotion != PieceType::kNone) {
    pt = PieceType::kPawn;
  }

  pieces_[static_cast<size_t>(pt)] ^= from_bb | to_bb;
  colors_[us] ^= from_bb | to_bb;

  if (m.promotion != PieceType::kNone) {
    pieces_[static_cast<size_t>(m.promotion)] ^= to_bb;
    pieces_[static_cast<size_t>(PieceType::kPawn)] ^= to_bb;
  }

  if (undo.captured_piece != PieceType::kNone) {
    if (pt == PieceType::kPawn && m.to == undo.old_en_passant_square) {
      const int ep_pawn_sq = undo.old_en_passant_square + (is_white ? -8 : 8);
      pieces_[static_cast<size_t>(PieceType::kPawn)] ^= square_bb(ep_pawn_sq);
      colors_[them] ^= square_bb(ep_pawn_sq);
    } else {
      pieces_[static_cast<size_t>(undo.captured_piece)] ^= to_bb;
      colors_[them] ^= to_bb;
    }
  }

  if (pt == PieceType::kKing) {
    const uint8_t idx = castle_index[(m.from << 6) | m.to];
    if (idx != 0) {
      const auto& cr = castle_rook[idx];
      const Bitboard rook = square_bb(cr.rook_from);
      const Bitboard rook_to = square_bb(cr.rook_to);
      pieces_[static_cast<size_t>(PieceType::kRook)] ^= rook | rook_to;
      colors_[cr.color_idx] ^= rook | rook_to;
    }
  }

  castling_rights_ = undo.old_castling_rights;
  en_passant_square_ = undo.old_en_passant_square;
  halfmove_clock_ = undo.old_halfmove_clock;
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

ChessState ChessState::from_fen(std::string_view fen) {
  ChessState s;
  s.castling_rights_ = 0;
  s.en_passant_square_ = -1;
  s.halfmove_clock_ = 0;
  s.fullmove_number_ = 1;

  const auto* it = fen.begin();
  const auto* end = fen.end();

  for (int rank = 7; rank >= 0; --rank) {
    int file = 0;
    while (file < 8 && it != end) {
      const char c = *it++;
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
      const bool is_white = (c >= 'A' && c <= 'Z');
      const int sq = rank * 8 + file;
      s.pieces_[static_cast<size_t>(pt)] |= (Bitboard{1} << sq);
      s.colors_[is_white ? 0 : 1] |= (Bitboard{1} << sq);
      ++file;
    }
    if (it != end && *it == '/') {
      ++it;
    }
  }

  if (it != end && *it == ' ') {
    ++it;
  }

  if (it != end) {
    s.side_to_move_ = (*it == 'w') ? Color::kWhite : Color::kBlack;
  }
  ++it;
  if (it != end && *it == ' ') {
    ++it;
  }

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
  if (it != end && *it == ' ') {
    ++it;
  }

  if (it != end && *it != '-') {
    const int file = *it - 'a';
    ++it;
    const int rank = *it - '1';
    ++it;
    s.en_passant_square_ = rank * 8 + file;
  } else {
    ++it;
  }
  if (it != end && *it == ' ') {
    ++it;
  }

  int halfmove = 0;
  while (it != end && *it >= '0' && *it <= '9') {
    halfmove = halfmove * 10 + (*it - '0');
    ++it;
  }
  s.halfmove_clock_ = halfmove;
  if (it != end && *it == ' ') {
    ++it;
  }

  int fullmove = 0;
  while (it != end && *it >= '0' && *it <= '9') {
    fullmove = fullmove * 10 + (*it - '0');
    ++it;
  }
  s.fullmove_number_ = fullmove;

  return s;
}

// https://www.chessprogramming.org/Forsyth-Edwards_Notation
std::string ChessState::ToString() const {
  std::string fen;

  for (int rank = 7; rank >= 0; --rank) {
    if (rank < 7) {
      fen += '/';
    }
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      const int sq = (rank * 8) + file;
      PieceType pt = piece_type_at(sq);
      if (pt == PieceType::kNone) {
        ++empty;
      } else {
        if (empty > 0) {
          fen += static_cast<char>('0' + empty);
          empty = 0;
        }
        const bool is_white = (colors_[0] & (Bitboard{1} << sq)) != 0;
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
    if (empty > 0) {
      fen += static_cast<char>('0' + empty);
    }
  }

  fen += ' ';
  fen += (side_to_move_ == Color::kWhite) ? 'w' : 'b';

  fen += ' ';
  if (castling_rights_ == 0) {
    fen += '-';
  } else {
    if (castling_rights_ & 1) {
      fen += 'K';
    }
    if (castling_rights_ & 2) {
      fen += 'Q';
    }
    if (castling_rights_ & 4) {
      fen += 'k';
    }
    if (castling_rights_ & 8) {
      fen += 'q';
    }
  }

  fen += ' ';
  if (en_passant_square_ == -1) {
    fen += '-';
  } else {
    fen += static_cast<char>('a' + (en_passant_square_ % 8));
    fen += static_cast<char>('1' + (en_passant_square_ / 8));
  }

  if (halfmove_clock_ != 0 || fullmove_number_ != 0) {
    fen += ' ';
    fen += std::to_string(halfmove_clock_);
    if (fullmove_number_ != 0) {
      fen += ' ';
      fen += std::to_string(fullmove_number_);
    }
  }

  return fen;
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

bool ChessState::IsOver() const {
  ChessMove buf[ChessMove::kMaxMoves];
  return FillLegalMoves(buf, ChessMove::kMaxMoves) == 0;
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
  const int to_file = san[2].str()[0] - 'a';
  const int to_rank = san[3].str()[0] - '1';
  const int to = (8 * to_rank) + to_file;

  ChessMove buf[ChessMove::kMaxMoves];
  const int n = FillLegalMoves(buf, ChessMove::kMaxMoves);
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
//
// All add_*_moves_fast and FillLegalMovesImpl function bodies live in
// chess.hpp so the compiler can inline them into FillLegalMovesImpl and
// each other (the per-node hot path of perft). Only the public dispatcher
// remains here.

size_t ChessState::FillLegalMoves(ChessMove* buffer, size_t capacity) const {
  (void)capacity;
  if (side_to_move_ == Color::kWhite) {
    return FillLegalMovesImpl<true>(buffer);
  }
  return FillLegalMovesImpl<false>(buffer);
}
