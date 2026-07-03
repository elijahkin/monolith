#include "chess_state.hpp"

#include <string>

#include "gtest/gtest.h"

TEST(ChessStateTest, PerftSmall) {
  auto initial = ChessState::initial_position();

  // https://www.chessprogramming.org/Perft_Results
  EXPECT_EQ(initial.PerftFast(0), 1UL);
  EXPECT_EQ(initial.PerftFast(1), 20UL);
  EXPECT_EQ(initial.PerftFast(2), 400UL);
  EXPECT_EQ(initial.PerftFast(3), 8'902UL);
  EXPECT_EQ(initial.PerftFast(4), 197'281UL);
  EXPECT_EQ(initial.PerftFast(5), 4'865'609UL);
}

TEST(ChessStateTest, Perft6) {
  auto initial = ChessState::initial_position();
  EXPECT_EQ(initial.PerftFast(6), 119'060'324UL);
}

TEST(ChessStateTest, Perft7) {
  auto initial = ChessState::initial_position();
  EXPECT_EQ(initial.PerftFast(7), 3'195'901'860UL);
}

TEST(ChessStateTest, FenRoundTrip) {
  const std::string start_fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  auto s = ChessState::from_fen(start_fen);
  EXPECT_EQ(s.to_fen(), start_fen);
}

TEST(ChessStateTest, FenKiwipete) {
  const std::string kp =
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
  auto s = ChessState::from_fen(kp);
  EXPECT_EQ(s.to_fen(), kp);
}

TEST(ChessStateTest, FenNoCastling) {
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
  auto s = ChessState::from_fen(fen);
  EXPECT_EQ(s.to_fen(), fen);
}

TEST(ChessStateTest, FenEnPassant) {
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
  auto s = ChessState::from_fen(fen);
  EXPECT_EQ(s.to_fen(), fen);
}

TEST(ChessStateTest, FenInitialPosition) {
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  auto from_fen = ChessState::from_fen(fen);
  auto from_init = ChessState::initial_position();
  EXPECT_EQ(from_fen.to_fen(), from_init.to_fen());
}
