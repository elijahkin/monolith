#include <cstddef>
#include <optional>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

#include "../genetic_operators.hpp"

class PermutationState {
 public:
  explicit PermutationState(std::vector<size_t> tour)
      : tour_(std::move(tour)) {}

  friend PermutationState crossover(const PermutationState& mother,
                                    const PermutationState& father,
                                    std::mt19937& rng) {
    return PermutationState(order_crossover(mother.tour_, father.tour_, rng));
  }

  friend void mutate(PermutationState& state, std::mt19937& rng) {
    inversion_mutate(state.tour_, rng);
    state.fitness_.reset();
  }

  [[nodiscard]] const std::vector<size_t>& tour() const { return tour_; }

  [[nodiscard]] double fitness() const { return *fitness_; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const PermutationState& state) {
    os << "[";
    for (size_t i = 0; i < state.tour_.size(); ++i) {
      os << state.tour_[i] << (i + 1 < state.tour_.size() ? ", " : "");
    }
    return os << "]";
  }

  friend class TspEvaluator;

 private:
  std::vector<size_t> tour_;
  mutable std::optional<double> fitness_;
};
