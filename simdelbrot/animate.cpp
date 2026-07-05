// Usage: bazel run //simdelbrot:animate simdelbrot/animation

#include <filesystem>
#include <string>
#include <string_view>

#include "../util/io.hpp"
#include "mandelbrot.hpp"

int main(int argc, char* argv[]) {
  const std::filesystem::path path = io::ResolveOutputPath(
      argc > 1 ? std::string_view{argv[1]} : std::string_view{});

  auto plotter = mandelbrot::Plotter(1920, 1080);

  // Rendering each frame of an animation
  double apothem = 3;
  for (int i = 0; i < 16080; ++i) {
    plotter.Plot(-0.77568377, 0.13646737, apothem, 800, std::to_string(i),
                 path.string());
    apothem *= 0.998929882193;
  }
  return 0;
}
