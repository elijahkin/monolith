#include "taxicab_state.hpp"

#include <cstddef>
#include <cstdint>

#include "collision_generator.hpp"
#include "gtest/gtest.h"

namespace {

using Power = __uint128_t;
using Base = size_t;
using State = SumOfTwoPowersState<Power, Base>;

template <typename S>
CollisionGenerator<S> MakeGenerator(size_t k, size_t num_ways, Power limit) {
  return CollisionGenerator<S>(num_ways, S::seed(k, limit));
}

TEST(TaxicabStateTest, SumOfTwoSquaresInTwoWays) {
  auto generator = MakeGenerator<State>(/*k=*/2, /*num_ways=*/2, /*limit=*/100);

  // The smallest such number is 50 = 1^2 + 7^2 = 5^2 + 5^2.
  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  const auto& value = result.value();
  EXPECT_EQ(static_cast<uint64_t>(value.key), 50UL);
}

TEST(TaxicabStateTest, SumOfTwoCubesInTwoWays) {
  auto generator =
      MakeGenerator<State>(/*k=*/3, /*num_ways=*/2, /*limit=*/12000);

  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  const auto& smallest = result.value();
  EXPECT_EQ(static_cast<uint64_t>(smallest.key), 1729UL);

  auto next_result = generator.next();
  ASSERT_TRUE(next_result.has_value());
  const auto& next_smallest = next_result.value();
  EXPECT_EQ(static_cast<uint64_t>(next_smallest.key), 4104UL);

  // The next smallest is 13832.
  auto done = generator.next();
  ASSERT_FALSE(done.has_value());
}

TEST(TaxicabStateTest, SumOfTwoCubesInThreeWays) {
  auto generator =
      MakeGenerator<State>(/*k=*/3, /*num_ways=*/3, /*limit=*/100'000'000);

  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  const auto& smallest = result.value();
  EXPECT_EQ(static_cast<uint64_t>(smallest.key), 87'539'319UL);

  auto done = generator.next();
  ASSERT_FALSE(done.has_value());
}

TEST(TaxicabStateTest, DifferenceOfTwoCubesInThreeWays) {
  using DiffState = DifferenceOfTwoPowersState<Power, Base>;
  auto generator =
      MakeGenerator<DiffState>(/*k=*/3, /*num_ways=*/3, /*limit=*/40'000);

  // The smallest such number is 3367 = 15^3 - 2^3 = 34^3 - 33^3 = 16^3 - 9^3.
  // Note that although the number itself is 3367, we can't find with a limit
  // under 34^3 = 39304. Philosophically, should limit be a property of the
  // number or the components?
  auto result = generator.next();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(static_cast<uint64_t>(result->key), 3367UL);

  auto done = generator.next();
  ASSERT_FALSE(done.has_value());
}

}  // namespace
