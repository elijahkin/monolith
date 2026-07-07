#include <immintrin.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
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
  [[nodiscard]] std::string to_fen() const;

  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] std::optional<ChessMove> Parse(const std::string& input) const;
  [[nodiscard]] size_t FillLegalMoves(ChessMove* buffer, size_t capacity) const;
  [[nodiscard]] bool IsOver() const;
  void RecordMove(const ChessMove& move);

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
    constexpr size_t kByI = static_cast<size_t>(kBy);
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
    constexpr size_t kThemI = static_cast<size_t>(kThem);
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

    Bitboard rook_pinners = rook_targets(king_sq, them_bb) & enemy_rooks;
    Bitboard bishop_pinners = bishop_targets(king_sq, them_bb) & enemy_bishops;

    Bitboard pinners = rook_pinners | bishop_pinners;
    Bitboard pinned = 0;

    while (pinners) {
      const int psq = std::countr_zero(pinners);
      pinners &= pinners - 1;
      Bitboard between = between_bb(king_sq, psq);
      Bitboard friends = between & us_bb;
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
    Bitboard occ = occupied();
    Bitboard them_bb = them();
    constexpr int kPushDir = kWhite ? 8 : -8;
    constexpr int kStartRank = kWhite ? 1 : 6;
    constexpr int kPromoRank = kWhite ? 6 : 1;
    constexpr int kAttackIdx = kWhite ? 0 : 1;

    while (pawns) {
      const int sq = std::countr_zero(pawns);
      pawns &= pawns - 1;
      const int r = sq / 8;
      const bool pinned = (pinned_bb & square_bb(sq)) != 0;
      Bitboard pin_ray = pinned && pin_rays ? pin_rays[sq] : ~Bitboard{0};

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
  [[nodiscard]] const char* to_unicode(int sq) const;
  [[nodiscard]] std::string get_algebraic_notation(const ChessMove& move) const;

  std::array<Bitboard, 7> pieces_{};
  std::array<Bitboard, 2> colors_{};
  Color side_to_move_ = Color::kWhite;
  uint8_t castling_rights_ = 0xF;
  int en_passant_square_ = -1;
  int halfmove_clock_ = 0;
  int fullmove_number_ = 1;

  std::vector<std::string> history_;
};
