#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "../states/permutation_state.hpp"

class TspEvaluator {
 public:
  struct City {
    double x, y;
  };

  explicit TspEvaluator(std::vector<City> cities)
      : cities_(std::move(cities)) {}

  template <std::ranges::range R>
  void evaluate_all(R& population) const {
    for (auto& state : population) {
      state.fitness_ = compute(state);
    }
  }

 private:
  [[nodiscard]] double compute(const PermutationState& state) const {
    const auto& tour = state.tour();
    double total = 0;
    for (size_t i = 0; i < tour.size(); ++i) {
      const City& a = cities_[tour[i]];
      const City& b = cities_[tour[(i + 1) % tour.size()]];
      total += distance(a, b);
    }
    return total;
  }

  static double distance(const City& city1, const City& city2) {
    double dx = city1.x - city2.x;
    double dy = city1.y - city2.y;
    return std::sqrt((dx * dx) + (dy * dy));
  }

  std::vector<City> cities_;
};
