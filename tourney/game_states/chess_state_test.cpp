#include "chess_state.hpp"

#include "../tree_searches/perft.hpp"
#include "gtest/gtest.h"

TEST(ChessStateTest, MovegenPerft) {
  auto s = ChessState::initial_position();

  Perft<ChessState> perft;

  // https://www.chessprogramming.org/Perft_Results
  EXPECT_EQ(perft.search(s, 0), 1UL);
  EXPECT_EQ(perft.search(s, 1), 20UL);
  EXPECT_EQ(perft.search(s, 2), 400UL);
  EXPECT_EQ(perft.search(s, 3), 8902UL);
  EXPECT_EQ(perft.search(s, 4), 197281UL);
  EXPECT_EQ(perft.search(s, 5), 4865609UL);
  // EXPECT_EQ(perft.search(s, 6), 119060324UL);
}
