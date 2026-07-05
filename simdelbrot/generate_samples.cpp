// Usage: bazel run //simdelbrot:generate_samples simdelbrot/samples

#include <filesystem>
#include <string>
#include <string_view>

#include "../util/io.hpp"
#include "mandelbrot.hpp"

int main(int argc, char* argv[]) {
  const std::filesystem::path path = io::ResolveOutputPath(
      argc > 1 ? std::string_view{argv[1]} : std::string_view{});

  auto plotter = mandelbrot::Plotter(2560, 1600);

  plotter.Plot(-0.6, 0, 2, 300, "mandelbrot", path.string());
  plotter.Plot(-0.77568377, 0.13646737, 1e-7, 1000, "misiurewicz",
               path.string());
  plotter.Plot(0.001643721971153, -0.822467633298876, 2e-11, 1600, "solar",
               path.string());
  return 0;
}
