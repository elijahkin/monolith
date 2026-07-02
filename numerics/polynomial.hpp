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

  friend bool operator==(const Polynomial& lhs, const Polynomial& rhs) {
    return lhs.coeffs_ == rhs.coeffs_;
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
    return *this;
  }

  // TODO We should eventually use `fft::fft` for this.
  Polynomial& operator*=(const Polynomial& rhs) {
    // TODO
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
    Polynomial result{};
    // TODO
    return result;
  }

  // TODO root computation?

  // ================================= Output =================================

  [[nodiscard]] std::string to_string() const {
    // TODO
    return "TODO";
  }

  friend std::ostream& operator<<(std::ostream& os, const Polynomial& p) {
    return os << p.to_string();
  }

 private:
  // Coeffificents stored lowest degree first
  std::vector<T> coeffs_;

  // TODO Should we have a `normalize` function?

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
