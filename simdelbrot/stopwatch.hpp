#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

// TODO This probably does not belong inside simdelbrot
class Stopwatch {
 public:
  Stopwatch() = default;

  void Lap() { time_points_.push_back(std::chrono::steady_clock::now()); }

  std::string Print() {
    std::string result;
    for (size_t i = 0; i < time_points_.size() - 1; ++i) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              time_points_[i + 1] - time_points_[i])
              .count();
      result += std::to_string(elapsed);
      result += "ms ";
    }
    return result;
  }

 private:
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_points_;
};
