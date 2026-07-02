#include "chess_state.hpp"

#include "gtest/gtest.h"

TEST(ChessStateTest, MovegenPerft) {
  auto initial = ChessState::initial_position();

  // https://www.chessprogramming.org/Perft_Results
  EXPECT_EQ(initial.Perft(0), 1UL);
  EXPECT_EQ(initial.Perft(1), 20UL);
  EXPECT_EQ(initial.Perft(2), 400UL);
  EXPECT_EQ(initial.Perft(3), 8902UL);
  EXPECT_EQ(initial.Perft(4), 197281UL);
  EXPECT_EQ(initial.Perft(5), 4865609UL);
  // EXPECT_EQ(initial.Perft(6), 119060324UL);
}
