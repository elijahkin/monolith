// Usage: generate_samples [output_directory]
// Example: ./generate_samples /tmp/mandelbrot_output

#include <cstdlib>
#include <filesystem>
#include <string>

#include "mandelbrot.hpp"

int main(int argc, char *argv[]) {
  // TODO I wonder if this should go into some utility function? It's currently
  // duplicated in animate.cpp and seems broadly useful.
  const char *workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  const std::filesystem::path output_path =
      argc > 1    ? std::filesystem::path(argv[1])
      : workspace ? std::filesystem::path(workspace)
                  : std::filesystem::current_path();
  const std::string path =
      (workspace && output_path.is_relative())
          ? (std::filesystem::path(workspace) / output_path).string()
          : output_path.string();
  std::filesystem::create_directories(path);

  auto plotter = mandelbrot::Plotter(2560, 1600);

  plotter.Plot(-0.6, 0, 2, 300, "mandelbrot", path);
  plotter.Plot(-0.77568377, 0.13646737, 1e-7, 1000, "misiurewicz", path);
  plotter.Plot(0.001643721971153, -0.822467633298876, 2e-11, 1600, "solar",
               path);
  return 0;
}
