#include <immintrin.h>
#include <png.h>
#include <pngconf.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include "stopwatch.hpp"

namespace mandelbrot {

using std::numbers::ln2;

using T = double;

const size_t kBatchPixels = 512 / (8 * sizeof(T));

class Plotter {
 public:
  Plotter(size_t width, size_t height)
      : width_(width),
        height_(height),
        num_pixels_(width * height),
        aspect_ratio_(static_cast<double>(height) / width) {
    real_.resize(num_pixels_);
    imag_.resize(num_pixels_);
    escape_iter_.resize(num_pixels_);
    rgb_.resize(3 * num_pixels_);
  }

  static void HslToRgb(const std::vector<double>& hsl,
                       std::vector<uint8_t>& rgb, size_t num_pixels) {
    double r, g, b;  // NOLINT

    for (size_t i = 0; i < num_pixels; ++i) {
      const double h = hsl[(3 * i) + 0];
      const double s = hsl[(3 * i) + 1];
      const double l = hsl[(3 * i) + 2];

      const double chroma = (1 - abs((2 * l) - 1)) * s;
      const double x = chroma * (1 - abs(fmod(h / 60, 2) - 1));
      const double m = l - (chroma / 2);

      switch (int(h / 60)) {
        case 0:
          r = chroma;
          g = x;
          b = 0;
          break;
        case 1:
          r = x;
          g = chroma;
          b = 0;
          break;
        case 2:
          r = 0;
          g = chroma;
          b = x;
          break;
        case 3:
          r = 0;
          g = x;
          b = chroma;
          break;
        case 4:
          r = x;
          g = 0;
          b = chroma;
          break;
        case 5:
          r = chroma;
          g = 0;
          b = x;
          break;
        default:
          r = g = b = 0;
          break;
      }
      rgb[(3 * i) + 0] = static_cast<uint8_t>((r + m) * 255);
      rgb[(3 * i) + 1] = static_cast<uint8_t>((g + m) * 255);
      rgb[(3 * i) + 2] = static_cast<uint8_t>((b + m) * 255);
    }
  }

  static void CreatePallete(std::vector<uint8_t>& palette, size_t max_iter) {
    std::vector<double> hsl(3 * max_iter);

    for (size_t i = 0; i < max_iter; ++i) {
      hsl[(3 * i) + 0] =
          fmod(powf((static_cast<float>(i) / max_iter) * 360, 1.5), 360);
      hsl[(3 * i) + 1] = 0.5;
      hsl[(3 * i) + 2] = 0.5;  // (double) i / max_iter;
    }

    HslToRgb(hsl, palette, max_iter);
  }

  static double Lerp(double a, double b, double t) { return a + (t * (b - a)); }

  void SavePng(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "wb");
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                              nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_set_IHDR(png, info, width_, height_, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> row_pointers(height_);
    for (size_t i = 0; i < height_; ++i) {
      row_pointers[i] = &rgb_[i * width_ * 3];
    }
    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
  }

  void Plot(T center_real, T center_imag, T apothem, size_t max_iter,
            const std::string& name, const std::string& output_path = ".") {
    // Compute the values needed for initialization
    const T min_real = center_real - apothem;
    const T max_imag = center_imag + (apothem * aspect_ratio_);
    const T real_step = 2 * apothem / (width_ - 1);
    const T imag_step = 2 * apothem / (height_ - 1) * aspect_ratio_;

    // Create the palette that will be used to look up colors for the points
    std::vector<uint8_t> palette(3 * max_iter);
    CreatePallete(palette, max_iter);

    // Time the main compute loop
    auto sw = Stopwatch();
    sw.Lap();

#pragma omp parallel for
    for (size_t i = 0; i < num_pixels_; ++i) {
      // Initialize the real and imag values
      const size_t row = i / width_;
      const size_t col = i % width_;
      imag_[i] = max_imag - static_cast<T>(row) * imag_step;
      real_[i] = min_real + static_cast<T>(col) * real_step;
    }
    sw.Lap();

    // Defining constants used in the loop below
    const __m512d zero = _mm512_setzero_pd();
    const __m512d four = _mm512_set1_pd(4);
    const __m512d MAX_ITER = _mm512_set1_pd(max_iter);

#pragma omp parallel for
    for (size_t i = 0; i < num_pixels_; i += kBatchPixels) {
      // Loading real and imaginary values from memory
      __m512d c_real = _mm512_loadu_pd(&real_[i]);
      __m512d c_imag = _mm512_loadu_pd(&imag_[i]);
      __m512d z_real = zero;
      __m512d z_imag = zero;
      __m512d ESCAPE_ITER = MAX_ITER;

      for (size_t iter = 0; iter < max_iter; ++iter) {
        // Computing the new real values
        const __m512d z_real_tmp = _mm512_fmsub_pd(
            z_real, z_real, _mm512_fmsub_pd(z_imag, z_imag, c_real));

        // Computing the new imaginary values
        const __m512d z_imag_tmp = _mm512_fmadd_pd(
            z_real, z_imag, _mm512_fmadd_pd(z_real, z_imag, c_imag));

        // Checking which points escaped
        const __m512d z_abs_sq =
            _mm512_fmadd_pd(z_real, z_real, _mm512_mul_pd(z_imag, z_imag));
        const __mmask8 escaped = _mm512_cmp_pd_mask(z_abs_sq, four, _CMP_GE_OQ);
        const __m512d this_iter = _mm512_set1_pd(iter);
        const __m512d blend =
            _mm512_mask_blend_pd(escaped, MAX_ITER, this_iter);
        ESCAPE_ITER = _mm512_min_pd(ESCAPE_ITER, blend);

        // Updating points with their new values, keeping escaped points fixed
        c_real = _mm512_mask_blend_pd(escaped, c_real, z_real);
        c_imag = _mm512_mask_blend_pd(escaped, c_imag, z_imag);
        z_real = _mm512_mask_blend_pd(escaped, z_real_tmp, zero);
        z_imag = _mm512_mask_blend_pd(escaped, z_imag_tmp, zero);
      }
      _mm512_storeu_pd(&escape_iter_[i], ESCAPE_ITER);
      _mm512_storeu_pd(&real_[i], c_real);
      _mm512_storeu_pd(&imag_[i], c_imag);
    }
    sw.Lap();

#pragma omp parallel for
    for (size_t i = 0; i < num_pixels_; ++i) {
      // Color the points according to their escape iteration
      const double log_zn = log(static_cast<double>((real_[i] * real_[i]) +
                                                    (imag_[i] * imag_[i]))) /
                            2;
      const double nu = log(log_zn / ln2) / ln2;
      escape_iter_[i] += 1 - nu;

      const auto iter = static_cast<size_t>(escape_iter_[i]);
      if (iter < max_iter) {
        rgb_[(3 * i) + 0] = static_cast<uint8_t>(
            Lerp(palette[(3 * iter) + 0], palette[(3 * iter) + 3],
                 fmod(escape_iter_[i], 1)));
        rgb_[(3 * i) + 1] = static_cast<uint8_t>(
            Lerp(palette[(3 * iter) + 1], palette[(3 * iter) + 4],
                 fmod(escape_iter_[i], 1)));
        rgb_[(3 * i) + 2] = static_cast<uint8_t>(
            Lerp(palette[(3 * iter) + 2], palette[(3 * iter) + 5],
                 fmod(escape_iter_[i], 1)));
      } else {
        rgb_[(3 * i) + 0] = 0;
        rgb_[(3 * i) + 1] = 0;
        rgb_[(3 * i) + 2] = 0;
      }
    }
    sw.Lap();

    std::cout << "Finished " << name << ": " << sw.Print() << '\n';
    SavePng(output_path + "/" + name + ".png");
  }

 private:
  std::vector<T> real_;
  std::vector<T> imag_;
  std::vector<double> escape_iter_;
  std::vector<uint8_t> rgb_;

  size_t width_;
  size_t height_;
  size_t num_pixels_;
  double aspect_ratio_;
};

}  // namespace mandelbrot
