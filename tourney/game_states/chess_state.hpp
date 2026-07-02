#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

class ChessState {
 public:
  enum class PieceType : uint8_t {
    kNone,
    kPawn,
    kKnight,
    kBishop,
    kRook,
    kQueen,
    kKing
  };

  struct Move {
    int from;
    int to;
    PieceType promotion;
  };

  // Stores information needed to undo a move
  struct MoveUndo {
    PieceType captured_piece;
    uint8_t old_castling_rights;
    int old_en_passant_square;
    int old_halfmove_clock;
  };

  [[nodiscard]] bool is_terminal() const { return legal_moves().empty(); }

  [[nodiscard]] double evaluate() const;
  std::vector<Move> legal_moves() const;

  MoveUndo make_move(Move m);
  void unmake_move(Move m, const MoveUndo& undo);

  static ChessState initial_position();

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
  };

  static AttackTables compute_attack_tables();
  inline static const AttackTables kAttacks = compute_attack_tables();

  Bitboard sliding_targets(Bitboard occ, int sq, int dir) const;
  Bitboard rook_targets(int sq, Bitboard occ) const {
    return sliding_targets(occ, sq, 8) | sliding_targets(occ, sq, -8) |
           sliding_targets(occ, sq, 1) | sliding_targets(occ, sq, -1);
  }
  Bitboard bishop_targets(int sq, Bitboard occ) const {
    return sliding_targets(occ, sq, 9) | sliding_targets(occ, sq, -9) |
           sliding_targets(occ, sq, 7) | sliding_targets(occ, sq, -7);
  }
  Bitboard queen_targets(int sq, Bitboard occ) const {
    return rook_targets(sq, occ) | bishop_targets(sq, occ);
  }

  bool is_attacked(int sq, Color by) const;
  bool in_check() const {
    int king_sq =
        std::countr_zero(pieces_[static_cast<size_t>(PieceType::kKing)] & us());
    return is_attacked(
        king_sq, static_cast<Color>(1 - static_cast<size_t>(side_to_move_)));
  }

  void add_pawn_moves(std::vector<Move>& moves) const;
  void add_knight_moves(std::vector<Move>& moves) const;
  void add_king_moves(std::vector<Move>& moves) const;
  void add_bishop_moves(std::vector<Move>& moves) const;
  void add_rook_moves(std::vector<Move>& moves) const;
  void add_queen_moves(std::vector<Move>& moves) const;

  template <typename F>
  void add_piece_moves(std::vector<Move>& moves, Bitboard pieces_bb,
                       F&& get_targets) const {
    while (pieces_bb) {
      int sq = std::countr_zero(pieces_bb);
      pieces_bb &= pieces_bb - 1;
      Bitboard targets = get_targets(sq) & ~us();
      while (targets) {
        int t = std::countr_zero(targets);
        targets &= targets - 1;
        moves.push_back({sq, t, PieceType::kNone});
      }
    }
  }

  std::vector<Move> pseudo_legal_moves() const;

  Bitboard occupied() const { return colors_[0] | colors_[1]; }
  Bitboard us() const { return colors_[static_cast<size_t>(side_to_move_)]; }
  Bitboard them() const {
    return colors_[1 - static_cast<size_t>(side_to_move_)];
  }

  std::array<Bitboard, 7> pieces_{};
  std::array<Bitboard, 2> colors_{};
  Color side_to_move_ = Color::kWhite;
  uint8_t castling_rights_ = 0xF;
  int en_passant_square_ = -1;
  int halfmove_clock_ = 0;
  int fullmove_number_ = 1;
};
