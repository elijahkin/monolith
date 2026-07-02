#include <cstdint>
#include <optional>
#include <string>

// TODO define these in terms of macros

// Underlying type of uint8_t allows for up to 256 distinct opcodes; increase if
// more should ever be needed
enum Opcode : uint8_t {
  kAbs,
  kAcos,
  kAcosh,
  kAdd,
  kAsin,
  kAsinh,
  kAtan,
  kAtanh,
  kAtan2,
  kCbrt,
  kConstant,
  kCos,
  kCosh,
  kDivide,
  kErf,
  kExp,
  kGamma,
  kLog,
  kMaximum,
  kMinimum,
  kMultiply,
  kNegate,
  kParameter,
  kPower,
  kRng,
  kSin,
  kSinh,
  kSqrt,
  kSubtract,
  kTan,
  kTanh
};

std::string opcode_to_string(Opcode opcode) {
  switch (opcode) {
    case kAbs:
      return "abs";
    case kAcos:
      return "acos";
    case kAcosh:
      return "acosh";
    case kAdd:
      return "add";
    case kAsin:
      return "asin";
    case kAsinh:
      return "asinh";
    case kAtan:
      return "atan";
    case kAtanh:
      return "atanh";
    case kAtan2:
      return "atan2";
    case kCbrt:
      return "cbrt";
    case kConstant:
      return "constant";
    case kCos:
      return "cos";
    case kCosh:
      return "cosh";
    case kDivide:
      return "divide";
    case kErf:
      return "erf";
    case kExp:
      return "exp";
    case kGamma:
      return "gamma";
    case kLog:
      return "log";
    case kMaximum:
      return "maximum";
    case kMinimum:
      return "minimum";
    case kMultiply:
      return "multiply";
    case kNegate:
      return "negate";
    case kParameter:
      return "parameter";
    case kPower:
      return "power";
    case kRng:
      return "rng";
    case kSin:
      return "sin";
    case kSinh:
      return "sinh";
    case kSqrt:
      return "sqrt";
    case kSubtract:
      return "subtract";
    case kTan:
      return "tan";
    case kTanh:
      return "tanh";
  }
}

int Arity(Opcode opcode) {
  switch (opcode) {
    case kConstant:
    case kParameter:
    case kRng:
      return 0;
    case kAbs:
    case kAcos:
    case kAcosh:
    case kAsin:
    case kAsinh:
    case kAtan:
    case kAtanh:
    case kCbrt:
    case kCos:
    case kCosh:
    case kErf:
    case kExp:
    case kGamma:
    case kLog:
    case kNegate:
    case kSin:
    case kSinh:
    case kSqrt:
    case kTan:
    case kTanh:
      return 1;
    case kAdd:
    case kAtan2:
    case kDivide:
    case kMaximum:
    case kMinimum:
    case kMultiply:
    case kPower:
    case kSubtract:
      return 2;
  }
}

bool IsEven(Opcode opcode) {
  switch (opcode) {
    case kAbs:
    case kCos:
    case kCosh:
      return true;
    default:
      return false;
  }
}

bool IsOdd(Opcode opcode) {
  switch (opcode) {
    case kAsin:
    case kAsinh:
    case kAtan:
    case kAtanh:
    case kCbrt:
    case kErf:
    case kNegate:
    case kSin:
    case kSinh:
    case kTan:
    case kTanh:
      return true;
    default:
      return false;
  }
}

bool IsIncreasing(Opcode opcode) {
  switch (opcode) {
    case kAcosh:
    case kAsin:
    case kAsinh:
    case kAtan:
    case kAtanh:
    case kCbrt:
    case kErf:
    case kExp:
    case kLog:
    case kSqrt:
    case kSinh:
    case kTanh:
      return true;
    default:
      return false;
  }
}

bool IsDecreasing(Opcode opcode) {
  switch (opcode) {
    case kAcos:
    case kNegate:
      return true;
    default:
      return false;
  }
}

bool IsNonNegative(Opcode opcode) {
  switch (opcode) {
    case kAcos:
    case kAcosh:
    case kCosh:
    case kExp:
    case kSqrt:
      return true;
    // TODO case kConstant:
    default:
      return false;
  }
}

std::optional<Opcode> Inverse(Opcode opcode) {
  switch (opcode) {
    case kAsinh:
      return kSinh;
    case kAtanh:
      return kTanh;
    case kExp:
      return kLog;
    case kCos:
      return kAcos;
    case kCosh:
      return kAcosh;
    case kLog:
      return kExp;
    case kNegate:
      return kNegate;
    case kSin:
      return kAsin;
    case kSinh:
      return kAsinh;
    case kTan:
      return kAtan;
    case kTanh:
      return kAtanh;
    default:
      return std::nullopt;
  }
}

bool IsCommutative(Opcode opcode) {
  switch (opcode) {
    case kAdd:
    case kMaximum:
    case kMinimum:
    case kMultiply:
      return true;
    default:
      return false;
  }
}
