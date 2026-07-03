#include <immintrin.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../contracts.hpp"

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

class ChessState : public Game<ChessMove> {
 public:
  struct MoveUndo {
    PieceType captured_piece;
    uint8_t old_castling_rights;
    int old_en_passant_square;
    int old_halfmove_clock;
  };

  MoveUndo make_move(ChessMove m);
  void unmake_move(ChessMove m, const MoveUndo& undo);

  // Perft-hot variants that skip irrelevant bookkeeping (halfmove/fullmove)
  MoveUndo make_move_perft(ChessMove m);
  void unmake_move_perft(ChessMove m, const MoveUndo& undo);

  static ChessState initial_position();
  static ChessState from_fen(std::string_view fen);
  [[nodiscard]] std::string to_fen() const;

  void MakeMove(const ChessMove& move) override;
  void UnmakeMove(const ChessMove& move) override;
  [[nodiscard]] std::string ToString() const override;
  [[nodiscard]] std::optional<ChessMove> Parse(
      const std::string& input) const override;
  [[nodiscard]] size_t FillLegalMoves(ChessMove* buffer,
                                      size_t capacity) const override;
  [[nodiscard]] bool IsOver() const override;
  void RecordMove(const ChessMove& move);

  // Fast perft without vector allocations or virtual dispatch
  [[nodiscard]] size_t PerftFast(size_t depth);

  // Fill pre-allocated buffer with legal moves, returns count
  size_t legal_moves_fast(ChessMove* moves) const;

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

  static Bitboard rook_targets(int sq, Bitboard occ) {
    Bitboard idx = _pext_u64(occ, rook_masks_[sq]);
    return rook_table_[rook_offset_[sq] + idx];
  }
  static Bitboard bishop_targets(int sq, Bitboard occ) {
    Bitboard idx = _pext_u64(occ, bishop_masks_[sq]);
    return bishop_table_[bishop_offset_[sq] + idx];
  }
  static Bitboard queen_targets(int sq, Bitboard occ) {
    return rook_targets(sq, occ) | bishop_targets(sq, occ);
  }

  static Bitboard between_bb(int sq1, int sq2) {
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

  [[nodiscard]] bool is_attacked(int sq, Color by,
                                 Bitboard slider_occ = ~Bitboard{0},
                                 Bitboard ignore = 0) const {
    if (slider_occ == ~Bitboard{0}) {
      slider_occ = occupied();
    }

    Bitboard enemy = colors_[static_cast<size_t>(by)] & ~ignore;

    if (kAttacks.pawn_attacks[1 - static_cast<size_t>(by)][sq] &
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

  Bitboard compute_checkers(int king_sq, Color them) const;
  Bitboard compute_pinned(int king_sq, Color us, Color them,
                          Bitboard* pin_rays) const;

  void add_pawn_moves_fast(ChessMove*& out, Bitboard pinned_bb = 0,
                           const Bitboard* pin_rays = nullptr,
                           Bitboard target_mask = ~Bitboard{0}) const;
  void add_knight_moves_fast(ChessMove*& out, Bitboard pinned_bb = 0,
                             const Bitboard* pin_rays = nullptr,
                             Bitboard target_mask = ~Bitboard{0}) const;
  void add_king_moves_fast(ChessMove*& out, int king_sq, Bitboard occ) const;
  void add_bishop_moves_fast(ChessMove*& out, Bitboard occ,
                             Bitboard pinned_bb = 0,
                             const Bitboard* pin_rays = nullptr,
                             Bitboard target_mask = ~Bitboard{0}) const;
  void add_rook_moves_fast(ChessMove*& out, Bitboard occ,
                           Bitboard pinned_bb = 0,
                           const Bitboard* pin_rays = nullptr,
                           Bitboard target_mask = ~Bitboard{0}) const;
  void add_queen_moves_fast(ChessMove*& out, Bitboard occ,
                            Bitboard pinned_bb = 0,
                            const Bitboard* pin_rays = nullptr,
                            Bitboard target_mask = ~Bitboard{0}) const;
  void add_en_passant_fast(ChessMove*& out, int king_sq, Bitboard occ) const;

  template <typename F>
  void add_piece_moves_fast(ChessMove*& out, Bitboard pieces_bb,
                            F&& get_targets, PieceType pt,
                            Bitboard pinned_bb = 0,
                            const Bitboard* pin_rays = nullptr,
                            Bitboard target_mask = ~Bitboard{0}) const {
    Bitboard unpinned = pieces_bb & ~pinned_bb;
    while (unpinned) {
      int sq = std::countr_zero(unpinned);
      unpinned &= unpinned - 1;
      Bitboard targets = get_targets(sq) & ~us() & target_mask;
      while (targets) {
        int t = std::countr_zero(targets);
        targets &= targets - 1;
        *out++ = ChessMove(sq, t, PieceType::kNone, pt);
      }
    }
    if (!pin_rays) {
      return;
    }
    Bitboard pinned = pieces_bb & pinned_bb;
    while (pinned) {
      int sq = std::countr_zero(pinned);
      pinned &= pinned - 1;
      Bitboard targets = get_targets(sq) & ~us() & target_mask & pin_rays[sq];
      while (targets) {
        int t = std::countr_zero(targets);
        targets &= targets - 1;
        *out++ = ChessMove(sq, t, PieceType::kNone, pt);
      }
    }
  }

  Bitboard occupied() const { return colors_[0] | colors_[1]; }
  Bitboard us() const { return colors_[static_cast<size_t>(side_to_move_)]; }
  Bitboard them() const {
    return colors_[1 - static_cast<size_t>(side_to_move_)];
  }

  [[nodiscard]] PieceType piece_type_at(int sq) const;
  [[nodiscard]] int unicode_index(int sq) const;
  [[nodiscard]] std::string get_algebraic_notation(const ChessMove& move) const;

  std::array<Bitboard, 7> pieces_{};
  std::array<Bitboard, 2> colors_{};
  std::array<PieceType, 64> board_{};
  Color side_to_move_ = Color::kWhite;
  uint8_t castling_rights_ = 0xF;
  int en_passant_square_ = -1;
  int halfmove_clock_ = 0;
  int fullmove_number_ = 1;

  std::vector<MoveUndo> undo_stack_;
  std::vector<std::string> history_;
};
