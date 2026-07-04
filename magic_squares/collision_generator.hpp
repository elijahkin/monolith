#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <queue>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template <typename S>
concept CollisionState = requires(const S& cs, S s) {
  { cs.key() };
  requires std::equality_comparable<decltype(cs.key())>;
  requires std::totally_ordered<decltype(cs.key())>;
  { cs.witness() };
  { cs > cs } -> std::convertible_to<bool>;
  { s.spawn_successors() };
  requires std::ranges::input_range<decltype(s.spawn_successors())>;
  requires std::same_as<
      std::ranges::range_value_t<decltype(s.spawn_successors())>, S>;
};

template <typename T>
  requires std::is_arithmetic_v<T>
std::string to_string(const T& w) {
  return std::to_string(w);
}

template <typename T, typename U>
std::string to_string(const std::pair<T, U>& p) {
  return ::to_string(p.first) + ' ' + ::to_string(p.second);
}

namespace detail {
template <typename T>
concept Stringifiable = requires(const T& w) {
  { to_string(w) };
};
}  // namespace detail

template <typename Key, typename Witness>
struct CollisionResult {
  Key key;
  std::vector<Witness> witnesses;

  [[nodiscard]] std::string to_string() const
    requires detail::Stringifiable<Witness>
  {
    std::string s;
    bool first = true;
    for (const auto& w : witnesses) {
      if (!first) {
        s += ' ';
      }
      s += ::to_string(w);
      first = false;
    }
    return s;
  }
};

template <CollisionState S>
class CollisionGenerator {
 public:
  using Key = decltype(std::declval<const S&>().key());
  using Witness = decltype(std::declval<const S&>().witness());
  using Result = CollisionResult<Key, Witness>;

  template <std::ranges::input_range SeedRange>
    requires std::same_as<std::ranges::range_value_t<SeedRange>, S>
  CollisionGenerator(size_t num_ways, SeedRange&& seeds) : num_ways_(num_ways) {
    for (S s : std::forward<SeedRange>(seeds)) {
      pq_.push(std::move(s));
    }
  }

  std::optional<Result> next() {
    std::vector<Witness> current_witnesses;
    current_witnesses.reserve(10);

    while (!pq_.empty()) {
      Key current_key = pq_.top().key();
      current_witnesses.clear();

      while (!pq_.empty() && pq_.top().key() == current_key) {
        S top = pq_.top();
        pq_.pop();
        current_witnesses.push_back(top.witness());

        for (S& succ : top.spawn_successors()) {
          pq_.push(std::move(succ));
        }
      }

      if (current_witnesses.size() >= num_ways_) {
        return Result{current_key, std::move(current_witnesses)};
      }
    }
    return std::nullopt;
  }

 private:
  size_t num_ways_;
  std::priority_queue<S, std::vector<S>, std::greater<>> pq_;
};
