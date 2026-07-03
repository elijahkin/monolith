#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace io {

namespace internal {

void CreateDirectoriesOrFail(const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (!std::filesystem::is_directory(dir, ec)) {
    std::cerr << "Error: could not create output directory " << dir << ": "
              << ec.message() << '\n';
    exit(1);
  }
}

std::filesystem::path WorkspaceRoot() {
  const char* workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");

  if (workspace != nullptr) {
    return std::filesystem::path(workspace);
  }

  const std::filesystem::path dir = std::filesystem::current_path();
  std::cerr << "Warning: BUILD_WORKSPACE_DIRECTORY is unset; writing to " << dir
            << '\n';
  return dir;
}

std::filesystem::path GetWorkspaceDirectory() {
  const std::filesystem::path dir = internal::WorkspaceRoot() / "log";
  internal::CreateDirectoriesOrFail(dir);
  return dir;
}

}  // namespace internal

std::filesystem::path ResolveOutputPath(std::string_view user_arg) {
  if (user_arg.empty()) {
    return internal::GetWorkspaceDirectory();
  }

  std::filesystem::path outpath(user_arg);
  if (outpath.is_absolute()) {
    internal::CreateDirectoriesOrFail(outpath);
    return outpath;
  }
  const std::filesystem::path dir = internal::WorkspaceRoot() / outpath;
  internal::CreateDirectoriesOrFail(dir);
  return dir;
}

std::ofstream GetOutputFileStream(const std::filesystem::path& directory,
                                  const std::string& file) {
  const std::string outpath = (directory / file).string();
  std::ofstream outfile(outpath);

  if (outfile.is_open()) {
    std::cout << "Opened output file " << outpath << '\n';
  } else {
    std::cerr << "Error: could not open output file " << outpath << '\n';
    exit(1);
  }

  return outfile;
}

}  // namespace io
