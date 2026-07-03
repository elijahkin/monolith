#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <ostream>

namespace spla {

// Defines the requirements to be a valid entry type of a `Tensor`. In
// particular, the type must support typical arithmetic operations such as
// addition, multiplication, absolute value, etc.
template <typename T>
concept Arithmetic = requires(T a, T b) {
  a += b;
  a -= b;
  a *= b;
  abs(a);
  exp(a);
  pow(a, b);
};

// Implements a sparse tensor represented internally by a map `data_`. If an
// index is not present as a key, its entry is implicitly `default_value_`.
template <Arithmetic T, int64_t... shape>
class Tensor {
 public:
  using Index = std::array<int64_t, sizeof...(shape)>;

  // This is required by the conversion constructor.
  template <Arithmetic OtherT, int64_t... other_shape>
  friend class Tensor;

  ///////////////////////
  // Factory Functions //
  ///////////////////////

  static auto full(T default_value) {
    return Tensor<T, shape...>(default_value);
  }

  static auto zeros() { return full(0); }

  static auto ones() { return full(1); }

  //////////////////
  // Constructors //
  //////////////////

  Tensor(T default_value) : default_value_(default_value) {}  // NOLINT

  template <Arithmetic OtherT>
  explicit Tensor(const Tensor<OtherT, shape...> &other)
      : default_value_(static_cast<T>(other.default_value_)) {
    for (const auto &[key, val] : other.data_) {
      data_[key] = static_cast<T>(val);
    }
  }

  template <Arithmetic OtherT, typename UnaryOp>
  Tensor(UnaryOp op, const Tensor<OtherT, shape...> &other)
      : default_value_(op(other.default_value_)) {
    for (const auto &[key, val] : other.data_) {
      data_[key] = op(val);
    }
  }

  template <Arithmetic OtherT, typename BinaryOp>
  Tensor(BinaryOp op, const Tensor<OtherT, shape...> &lhs,
         const Tensor<OtherT, shape...> &rhs)
      : default_value_(op(lhs.default_value_, rhs.default_value_)) {
    for (const auto &[key, lhs_val] : lhs.data_) {
      if (auto it = rhs.data_.find(key); it != rhs.data_.end()) {
        // The key is in both lhs and rhs
        data_[key] = op(lhs_val, it->second);
      } else {
        // The key is only in lhs
        data_[key] = op(lhs_val, rhs.default_value_);
      }
    }
    for (const auto &[key, rhs_val] : rhs.data_) {
      if (auto it = lhs.data_.find(key); it == lhs.data_.end()) {
        // The key is only in rhs
        data_[key] = op(lhs.default_value_, rhs_val);
      }
    }
  }

  /////////////////////////
  // Subscript Operation //
  /////////////////////////

  // Handles reading and writing of tensor entries. This is the object returned
  // by `operator[]`, which is necessary to ensure that default values are not
  // accidently written to `data_`.
  class SubscriptProxy {
   public:
    SubscriptProxy(Tensor<T, shape...> *vec, Index idx)
        : vec_(vec), idx_(idx) {}

    // Handles tensor subscript assignment.
    SubscriptProxy &operator=(const T &val) {
      vec_->data_[idx_] = val;
      return *this;
    }

    // Casts a `SubscriptProxy` object to the entry type `T`. It searches for
    // the whether `idx_` is a key in `vec_.data_`. If so, it returns its
    // associated value, and `vec_.default_value_` otherwise.
    operator T() const {  // NOLINT
      constexpr auto kDims = std::array{shape...};
      for (size_t i = 0; i < kDims.size(); ++i) {
        if (idx_[i] < 0 || idx_[i] >= kDims[i]) {
          return T{};
        }
      }
      if (auto it = vec_->data_.find(idx_); it != vec_->data_.end()) {
        return it->second;
      }
      return vec_->default_value_;
    }

   private:
    Tensor<T, shape...> *vec_;
    Index idx_;
  };

  template <typename... Args>
  SubscriptProxy operator[](Args... idx) {
    return SubscriptProxy(this, {static_cast<int64_t>(idx)...});
  }

  //////////////////////////////////////
  // Modifying Elementwise Operations //
  //////////////////////////////////////

  auto &operator+=(const Tensor<T, shape...> &rhs) {
    return this->apply_binary_inplace([](T &a, const T &b) { a += b; }, rhs);
  }

  auto &operator-=(const Tensor<T, shape...> &rhs) {
    return this->apply_binary_inplace([](T &a, const T &b) { a -= b; }, rhs);
  }

  auto &operator*=(const Tensor<T, shape...> &rhs) {
    return this->apply_binary_inplace([](T &a, const T &b) { a *= b; }, rhs);
  }

  ///////////////////
  // Miscellaneous //
  ///////////////////

  friend std::ostream &operator<<(std::ostream &os,
                                  const Tensor<T, shape...> &vec) {
    return os << std::format("{}", vec.data_);
  }

  int64_t elements_in() const { return (1 * ... * shape); }

  int64_t sparsity() const { return data_.size(); }

  friend T reduce(const Tensor<T, shape...> &vec,
                  std::function<void(T &, const T &)> op) {
    T result = static_cast<T>(vec.default_value_ *
                              (vec.elements_in() - vec.sparsity()));
    for (const auto &[_, val] : vec.data_) {
      op(result, val);
    }
    return result;
  }

  friend auto pow(const Tensor<T, shape...> &lhs,
                  const Tensor<T, shape...> &rhs) {
    return Tensor<T, shape...>([](T a, T b) { return std::pow(a, b); }, lhs,
                               rhs);
  }

 private:
  std::map<Index, T> data_;
  T default_value_;

  auto &apply_binary_inplace(std::function<void(T &, const T &)> op,
                             const Tensor<T, shape...> &rhs) {
    for (auto &[key, lhs_val] : data_) {
      if (auto it = rhs.data_.find(key); it == rhs.data_.end()) {
        // The key is only in lhs
        op(lhs_val, rhs.default_value_);
      }
    }
    for (const auto &[key, rhs_val] : rhs.data_) {
      if (auto it = data_.find(key); it == data_.end()) {
        // The key is only in rhs
        data_[key] = default_value_;
      }
      op(data_[key], rhs_val);
    }
    op(default_value_, rhs.default_value_);
    return *this;
  }
};

//////////////////////////////////////////
// Non-modifying Elementwise Operations //
//////////////////////////////////////////

template <Arithmetic T, int64_t... shape>
auto abs(const Tensor<T, shape...> &vec) {
  return Tensor<T, shape...>([](T a) { return std::abs(a); }, vec);
}

template <Arithmetic T, int64_t... shape>
auto exp(const Tensor<T, shape...> &vec) {
  return Tensor<T, shape...>([](T a) { return std::exp(a); }, vec);
}

template <Arithmetic T, int64_t... shape>
auto operator+(const Tensor<T, shape...> &lhs, const Tensor<T, shape...> &rhs) {
  return Tensor<T, shape...>([](T a, T b) { return a + b; }, lhs, rhs);
}

template <Arithmetic T, int64_t... shape>
auto operator-(const Tensor<T, shape...> &lhs, const Tensor<T, shape...> &rhs) {
  return Tensor<T, shape...>([](T a, T b) { return a - b; }, lhs, rhs);
}

template <Arithmetic T, int64_t... shape>
auto operator*(const Tensor<T, shape...> &lhs, const Tensor<T, shape...> &rhs) {
  return Tensor<T, shape...>([](T a, T b) { return a * b; }, lhs, rhs);
}

template <Arithmetic T, int64_t... shape>
auto operator<(const Tensor<T, shape...> &lhs, const Tensor<T, shape...> &rhs) {
  return Tensor<bool, shape...>([](T a, T b) { return a < b; }, lhs, rhs);
}

template <Arithmetic T, int64_t... shape>
auto operator==(const Tensor<T, shape...> &lhs,
                const Tensor<T, shape...> &rhs) {
  return Tensor<bool, shape...>([](T a, T b) { return a == b; }, lhs, rhs);
}

///////////////////////
// Reduce Operations //
///////////////////////

template <int64_t... shape>
bool all(const Tensor<bool, shape...> &vec) {
  return reduce(vec, [](bool &a, const bool &b) { return a &= b; });
}

template <int64_t... shape>
bool any(const Tensor<bool, shape...> &vec) {
  return reduce(vec, [](bool &a, const bool &b) { a |= b; });
}

template <Arithmetic T, int64_t... shape>
T sum(const Tensor<T, shape...> &vec) {
  return reduce(vec, [](T &a, const T &b) { a += b; });
}

template <Arithmetic T, int64_t... shape>
T dot(const Tensor<T, shape...> &lhs, const Tensor<T, shape...> &rhs) {
  return sum(lhs * rhs);
}

template <Arithmetic T, int64_t... shape>
double norm(const Tensor<T, shape...> &vec, int ord) {
  return std::pow(sum(pow(abs(vec), ord)), 1.0 / ord);
}

}  // namespace spla
