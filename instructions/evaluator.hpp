#include <cmath>
#include <vector>

#include "instruction.hpp"

double Evaluate(Instruction* instruction) {
  std::vector<double> evaluated;
  for (auto* operand : instruction->operands()) {
    evaluated.push_back(Evaluate(operand));
  }

  switch (instruction->opcode()) {
    case kAbs:
      return abs(evaluated[0]);
    case kAcos:
      return acos(evaluated[0]);
    case kAcosh:
      return acosh(evaluated[0]);
    case kAdd:
      return evaluated[0] + evaluated[1];
    case kAsin:
      return asin(evaluated[0]);
    case kAsinh:
      return asinh(evaluated[0]);
    case kAtan:
      return atan(evaluated[0]);
    case kAtanh:
      return atanh(evaluated[0]);
    case kAtan2:
      return atan2(evaluated[0], evaluated[0]);
    case kCbrt:
      return cbrt(evaluated[0]);
    case kConstant:
      return static_cast<ConstantInstruction*>(instruction)->value();
    case kCos:
      return cos(evaluated[0]);
    case kCosh:
      return cosh(evaluated[0]);
    case kDivide:
      return evaluated[0] / evaluated[1];
    case kErf:
      return erf(evaluated[0]);
    case kExp:
      return exp(evaluated[0]);
    case kGamma:
      return tgamma(evaluated[0]);
    case kLog:
      return log(evaluated[0]);
    case kMaximum:
      return fmax(evaluated[0], evaluated[1]);
    case kMinimum:
      return fmin(evaluated[0], evaluated[1]);
    case kMultiply:
      return evaluated[0] * evaluated[1];
    case kNegate:
      return -evaluated[0];
    case kPower:
      return pow(evaluated[0], evaluated[1]);
    case kSin:
      return sin(evaluated[0]);
    case kSinh:
      return sinh(evaluated[0]);
    case kSqrt:
      return sqrt(evaluated[0]);
    case kSubtract:
      return evaluated[0] - evaluated[1];
    case kTan:
      return tan(evaluated[0]);
    case kTanh:
      return tanh(evaluated[0]);
    // TODO Eventually make sure all opcodes are implemented
    default:
      return 0;
  }
}
