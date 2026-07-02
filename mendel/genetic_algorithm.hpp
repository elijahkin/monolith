#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <ranges>

struct GeneticAlgorithmOptions {
  // Controls the stopping criterion for the genetic algorithm. Allows us to
  // easily run for a set number of generations, or until we reach fitness under
  // some threshold, or until we haven't improved in X generations.
  std::function<bool(size_t)> not_finished = [](size_t generation) {
    return generation < 50;
  };

  double mutation_rate = 0.05;

  size_t tournament_size = 3;

  // Controls the number of best individuals carried into the next generation.
  size_t elitism_count = 5;

  // Controls whether we log generational statistics.
  bool verbose = false;

  // We deliberately do not include the RNG device since multithreading requires
  // each thread to own its own device.
};

template <std::ranges::random_access_range R, typename Eval>
void run_genetic_algorithm(R& population, Eval& evaluator,
                           GeneticAlgorithmOptions options, std::mt19937& rng) {
  using StateT = std::ranges::range_value_t<R>;
  size_t n = population.size();

  R next_population = population;

  std::bernoulli_distribution mutate_chance(options.mutation_rate);

  size_t generation = 0;
  while (options.not_finished(generation)) {
    // Evaluate the fitness of every individual in the generation.
    evaluator.evaluate_all(population);

    // Partially sort the population so elite individuals are first.
    auto lower_is_better = [](const StateT& x, const StateT& y) {
      return x.fitness() < y.fitness();
    };
    std::ranges::partial_sort(population,
                              population.begin() + options.elitism_count,
                              lower_is_better);

    // Report generational statistics if desired.
    if (options.verbose) {
      std::cout << "Gen " << generation << ": best=" << population[0].fitness()
                << '\n';
    }

    // Carry over the desired amount of elites to the next generation.
    for (size_t i = 0; i < options.elitism_count && i < n; ++i) {
      next_population[i] = population[i];
    }

    // Parent select and breed for the rest of the next generation.
    for (size_t i = options.elitism_count; i < n; ++i) {
      const StateT& mother =
          tournament_select(population, options.tournament_size, rng);
      const StateT& father =
          tournament_select(population, options.tournament_size, rng);

      StateT child = crossover(mother, father, rng);
      if (mutate_chance(rng)) {
        mutate(child, rng);
      }
      next_population[i] = std::move(child);
    }

    // Ping-pong the generaiton buffers.
    std::swap(population, next_population);
    ++generation;
  }
}
