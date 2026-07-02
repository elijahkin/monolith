#include "number_theory.hpp"

#include "gtest/gtest.h"

TEST(NumberTheoryTest, SieveOfEratosthenes) {
  auto is_prime = nt::sieve_of_eratosthenes<10000>();
  EXPECT_EQ(is_prime.count(), 1229U);
}

TEST(NumberTheoryTest, ExtendedEuclideanAlgorithm) {
  auto [gcd, x, y] = nt::xgcd(15, 26);
  EXPECT_EQ(gcd, 1);
  EXPECT_EQ(x, 7);
  EXPECT_EQ(y, -4);
}

TEST(NumberTheoryTest, ChineseRemainderTheorem) {
  // A problem from the 5th-century book Sunzi Suanjing
  auto a = nt::crt<int>({{2, 3}, {3, 5}, {2, 7}});
  EXPECT_EQ(a, 23);
}

// TODO This fails to build right now?
// TEST(NumberTheoryTest, ModularPowers) {
//   EXPECT_EQ(nt::powmod<uint32_t>(3, 5, 7), 5);
//   EXPECT_EQ(nt::powmod<uint32_t>(10, 0, 7), 1);
// }
