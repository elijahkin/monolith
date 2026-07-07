#include "chess.hpp"

#include <string_view>

#include "../algorithms/perft.hpp"
#include "gtest/gtest.h"

// https://www.chessprogramming.org/Perft_Results

TEST(ChessStateTest, InitialPosition) {
  auto position = ChessState::initial_position();
  PerftSearch<ChessState> perft(position);

  EXPECT_EQ(perft.search(0), 1UL);
  EXPECT_EQ(perft.search(1), 20UL);
  EXPECT_EQ(perft.search(2), 400UL);
  EXPECT_EQ(perft.search(3), 8'902UL);
  EXPECT_EQ(perft.search(4), 197'281UL);
  EXPECT_EQ(perft.search(5), 4'865'609UL);
  EXPECT_EQ(perft.search(6), 119'060'324UL);
}

TEST(ChessStateTest, Position2) {
  constexpr std::string_view kFen =
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";

  auto position = ChessState::from_fen(kFen);
  EXPECT_EQ(position.ToString(), kFen);

  PerftSearch<ChessState> perft(position);

  EXPECT_EQ(perft.search(1), 48UL);
  EXPECT_EQ(perft.search(2), 2'039UL);
  EXPECT_EQ(perft.search(3), 97'862UL);
  EXPECT_EQ(perft.search(4), 4'085'603UL);
  EXPECT_EQ(perft.search(5), 193'690'690UL);
}

TEST(ChessStateTest, Position3) {
  constexpr std::string_view kFen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

  auto position = ChessState::from_fen(kFen);
  EXPECT_EQ(position.ToString(), kFen);

  PerftSearch<ChessState> perft(position);

  EXPECT_EQ(perft.search(1), 14UL);
  EXPECT_EQ(perft.search(2), 191UL);
  EXPECT_EQ(perft.search(3), 2'812UL);
  EXPECT_EQ(perft.search(4), 43'238UL);
  EXPECT_EQ(perft.search(5), 674'624UL);
  EXPECT_EQ(perft.search(6), 11'030'083UL);
}

TEST(ChessStateTest, Position4) {
  constexpr std::string_view kFen =
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

  auto position = ChessState::from_fen(kFen);
  EXPECT_EQ(position.ToString(), kFen);

  PerftSearch<ChessState> perft(position);

  EXPECT_EQ(perft.search(1), 6UL);
  EXPECT_EQ(perft.search(2), 264UL);
  EXPECT_EQ(perft.search(3), 9'467UL);
  EXPECT_EQ(perft.search(4), 422'333UL);
  EXPECT_EQ(perft.search(5), 15'833'292UL);
}

TEST(ChessStateTest, Position5) {
  constexpr std::string_view kFen =
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

  auto position = ChessState::from_fen(kFen);
  EXPECT_EQ(position.ToString(), kFen);

  PerftSearch<ChessState> perft(position);

  EXPECT_EQ(perft.search(1), 44UL);
  EXPECT_EQ(perft.search(2), 1'486UL);
  EXPECT_EQ(perft.search(3), 62'379UL);
  EXPECT_EQ(perft.search(4), 2'103'487UL);
  EXPECT_EQ(perft.search(5), 89'941'194UL);
}

TEST(ChessStateTest, Position6) {
  constexpr std::string_view kFen =
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 "
      "10";

  auto position = ChessState::from_fen(kFen);
  EXPECT_EQ(position.ToString(), kFen);

  PerftSearch<ChessState> perft(position);

  EXPECT_EQ(perft.search(1), 46UL);
  EXPECT_EQ(perft.search(2), 2'079UL);
  EXPECT_EQ(perft.search(3), 89'890UL);
  EXPECT_EQ(perft.search(4), 3'894'594UL);
  EXPECT_EQ(perft.search(5), 164'075'551UL);
}
