#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "evaluators/tsp_evaluator.hpp"
#include "genetic_algorithm.hpp"

int main() {
  // TODO Eventually parse in from the command line
  size_t population_size = 256;

  GeneticAlgorithmOptions options;
  options.verbose = true;

  std::mt19937 rng;

  // Create a random TSP evaluator.
  std::uniform_real_distribution<double> coord_dist(0.0, 100.0);
  std::vector<TspEvaluator::City> cities(10);
  for (auto& c : cities) {
    c.x = coord_dist(rng);
    c.y = coord_dist(rng);
  }
  TspEvaluator evaluator(std::move(cities));

  std::vector<size_t> base_tour(10);
  std::iota(base_tour.begin(), base_tour.end(), 0);

  // Initialize a random population
  std::vector<PermutationState> population;
  population.reserve(population_size);
  for (size_t i = 0; i < population_size; ++i) {
    std::ranges::shuffle(base_tour, rng);
    population.emplace_back(base_tour);
  }

  run_genetic_algorithm(population, evaluator, options, rng);

  // TODO Report the best solution found
  return 0;
}
