#include "taxicab_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "collision_generator.hpp"
#include "gtest/gtest.h"

namespace {

using Power = __uint128_t;
using Base = size_t;
using State = SumOfTwoPowersState<Power, Base>;
using Generator = CollisionGenerator<State>;

std::vector<std::pair<size_t, size_t>> NormalizePairs(
    std::vector<std::pair<size_t, size_t>> pairs) {
  for (auto& [a, b] : pairs) {
    if (a > b) {
      std::swap(a, b);
    }
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

Generator MakeGenerator(size_t k, size_t num_ways, Power limit) {
  auto seeds = State::seed(k, limit);
  return Generator(num_ways, seeds);
}

TEST(TaxicabStateTest, FindsRamanujan1729) {
  auto generator = MakeGenerator(/*k=*/3, /*num_ways=*/2, /*limit=*/2000);

  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  const auto& value = result.value();

  EXPECT_EQ(static_cast<uint64_t>(value.key), 1729ULL);

  const auto expected = NormalizePairs({{1, 12}, {9, 10}});
  const auto actual = NormalizePairs({value.witnesses[0], value.witnesses[1]});
  EXPECT_EQ(actual, expected);
}

TEST(TaxicabStateTest, FindsTaxicab4104) {
  // 4104 = 2^3 + 16^3 = 9^3 + 15^3 is the second taxicab number, so it must
  // follow 1729 in the stream emitted by `next()`.
  auto generator = MakeGenerator(/*k=*/3, /*num_ways=*/2, /*limit=*/5000);

  ASSERT_TRUE(generator.next().has_value());  // skip 1729

  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  const auto& value = result.value();
  EXPECT_EQ(static_cast<uint64_t>(value.key), 4104ULL);

  const auto expected = NormalizePairs({{2, 16}, {9, 15}});
  const auto actual = NormalizePairs({value.witnesses[0], value.witnesses[1]});
  EXPECT_EQ(actual, expected);
}

TEST(TaxicabStateTest, FindsSumsOfTwoSquares) {
  // k=2 asks for numbers expressible as a sum of two squares in two distinct
  // ways. The smallest such number is 50 = 1^2 + 7^2 = 5^2 + 5^2.
  auto generator = MakeGenerator(/*k=*/2, /*num_ways=*/2, /*limit=*/100);

  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  const auto& value = result.value();
  EXPECT_EQ(static_cast<uint64_t>(value.key), 50ULL);

  const auto expected = NormalizePairs({{1, 7}, {5, 5}});
  const auto actual = NormalizePairs({value.witnesses[0], value.witnesses[1]});
  EXPECT_EQ(actual, expected);
}

}  // namespace
