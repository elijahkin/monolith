#include <cstddef>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

// Order crossover (OX1): preserves a segment from the mother and fills the
// remaining positions from the father while maintaining relative order.
template <typename T>
T order_crossover(const T& mother, const T& father, std::mt19937& rng) {
  size_t n = mother.size();
  if (n < 2) {
    return mother;
  }

  std::uniform_int_distribution<size_t> dist(0, n - 1);
  size_t start = dist(rng);
  size_t end = dist(rng);
  if (start > end) {
    std::swap(start, end);
  }

  T child(n);
  std::vector<bool> taken(n, false);

  for (size_t i = start; i <= end; ++i) {
    child[i] = mother[i];
    taken[mother[i]] = true;
  }

  size_t child_pos = (end + 1) % n;
  for (size_t i = 0; i < n; ++i) {
    size_t father_idx = (end + 1 + i) % n;
    auto city = father[father_idx];
    if (!taken[city]) {
      child[child_pos] = city;
      taken[city] = true;
      child_pos = (child_pos + 1) % n;
    }
  }

  return child;
}

// Invert a partial list of permutation with periodic boundary conditions.
template <typename T>
void inversion_mutate(T& permutation, std::mt19937& rng) {
  size_t n = permutation.size();
  std::uniform_int_distribution<size_t> dist(0, n - 1);
  size_t i = dist(rng);
  size_t j = dist(rng);

  if (i == j) {
    return;
  }

  // Handle periodic boundaries
  size_t length = (j > i) ? (j - i + 1) : (n - i + j + 1);
  for (size_t k = 0; k < length / 2; ++k) {
    size_t a = (i + k) % n;
    size_t b = (i + length - 1 - k) % n;
    std::swap(permutation[a], permutation[b]);
  }
}

// Sample a number of individuals uniformly and return the fittest.
template <std::ranges::range R>
const std::ranges::range_value_t<R>& tournament_select(const R& population,
                                                       size_t tournament_size,
                                                       std::mt19937& rng) {
  std::uniform_int_distribution<size_t> dist(0, population.size() - 1);
  size_t best_idx = dist(rng);
  for (size_t k = 1; k < tournament_size; ++k) {
    size_t idx = dist(rng);
    if (population[idx].fitness() < population[best_idx].fitness()) {
      best_idx = idx;
    }
  }
  return population[best_idx];
}
