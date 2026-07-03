#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "algebraic_concepts.hpp"

namespace numerics {

// Univariate polynomial over an arbitrary ring using a vector of coefficients.
// The zero polynomial is represented by an empty vector.
template <RingElement T>
class Polynomial {
 public:
  Polynomial(std::initializer_list<T> coeffs) : coeffs_(coeffs) {}
  explicit Polynomial(std::vector<T> coeffs) : coeffs_(std::move(coeffs)) {}

  friend bool operator==(const Polynomial& lhs, const Polynomial& rhs) {
    return lhs.normalized() == rhs.normalized();
  }

  // TODO Spaceship operator? I'm not sure how useful it would be...

  // =============================== Accessors ================================

  [[nodiscard]] std::optional<size_t> degree() const {
    if (coeffs_.empty()) {
      return std::nullopt;
    }
    return coeffs_.size() - 1;
  }

  // Evaluate polynomial via Horner's method.
  [[nodiscard]] T operator()(T x) const { return horner(x); }

  // =============================== Arithmetic ===============================

  Polynomial& operator+=(const Polynomial& rhs) {
    if (rhs.coeffs_.size() > coeffs_.size()) {
      coeffs_.resize(rhs.coeffs_.size(), T(0));
    }
    for (size_t i = 0; i < rhs.coeffs_.size(); ++i) {
      coeffs_[i] += rhs.coeffs_[i];
    }
    normalize();
    return *this;
  }

  Polynomial& operator*=(const Polynomial& rhs) {
    if (coeffs_.empty() || rhs.coeffs_.empty()) {
      coeffs_.clear();
      return *this;
    }
    std::vector<T> result(coeffs_.size() + rhs.coeffs_.size() - 1, T(0));
    for (size_t i = 0; i < coeffs_.size(); ++i) {
      for (size_t j = 0; j < rhs.coeffs_.size(); ++j) {
        result[i + j] += coeffs_[i] * rhs.coeffs_[j];
      }
    }
    coeffs_ = std::move(result);
    normalize();
    return *this;
  }

  [[nodiscard]] friend Polynomial operator+(Polynomial lhs,
                                            const Polynomial& rhs) {
    return lhs += rhs;
  }

  [[nodiscard]] friend Polynomial operator*(Polynomial lhs,
                                            const Polynomial& rhs) {
    return lhs *= rhs;
  }

  // ================================ Calculus ================================

  Polynomial deriv() const {
    if (coeffs_.empty() || coeffs_.size() == 1) {
      return Polynomial{};
    }
    std::vector<T> result(coeffs_.size() - 1);
    for (size_t i = 1; i < coeffs_.size(); ++i) {
      result[i - 1] = coeffs_[i] * static_cast<T>(i);
    }
    return Polynomial(std::move(result));
  }

  // TODO root computation?

  // ================================= Output =================================

  [[nodiscard]] std::string to_string() const {
    if (coeffs_.empty()) {
      return "0";
    }
    std::string result;
    for (size_t i = coeffs_.size(); i-- > 0;) {
      if (coeffs_[i] == T(0)) {
        continue;
      }
      if (!result.empty()) {
        result += '+';
      }
      if (i == 0) {
        result += std::to_string(coeffs_[i]);
      } else {
        if (coeffs_[i] != T(1)) {
          result += std::to_string(coeffs_[i]);
        }
        result += 'x';
        if (i > 1) {
          result += '^' + std::to_string(i);
        }
      }
    }
    if (result.empty()) {
      return "0";
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const Polynomial& p) {
    return os << p.to_string();
  }

 private:
  // Coeffificents stored lowest degree first
  std::vector<T> coeffs_;

  void normalize() {
    while (!coeffs_.empty() && coeffs_.back() == T(0)) {
      coeffs_.pop_back();
    }
  }

  [[nodiscard]] std::vector<T> normalized() const {
    size_t last = coeffs_.size();
    while (last > 0 && coeffs_[last - 1] == T(0)) {
      --last;
    }
    return std::vector<T>(coeffs_.begin(), coeffs_.begin() + last);
  }

  T horner(T x) const {
    if (coeffs_.empty()) {
      return T(0);
    }

    T result = coeffs_.back();
    for (auto it = coeffs_.rbegin() + 1; it != coeffs_.rend(); ++it) {
      result = result * x + (*it);
    }
    return result;
  }
};

template <typename T>
Polynomial<T> lagrange_interp(const std::vector<T>& xs,
                              const std::vector<T>& ys) {
  // TODO
}

// TODO Does this need to be a separate class? How will it differ? Would it be
// more elegant to have a single class with specializations for different
// families of polynomials? I feel like instead of having this be a separate
// class, a polynomial should come equipped with a `Basis` enum class element?
template <typename T>
class ChebPoly {
 public:
  // TODO

 private:
  std::vector<T> cheb_coeffs_;

  T clenshaw(T x) const {
    // TODO Kind of like Horner?
  }
};

template <typename T>
ChebPoly<T> cheb_interp(const std::vector<T>& xs, const std::vector<T>& ys) {
  // TODO
}

}  // namespace numerics
