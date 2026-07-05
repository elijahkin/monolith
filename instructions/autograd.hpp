#include "instruction.hpp"

class Autograd {
 private:
  // The derivative of f(g(x)) is f'(g(x))*g'(x) by the chain rule; this
  // function accepts f(g(x)) and returns f'(g(x))
  static Instruction *ChainRuleHelper(Instruction *instruction) {
    const Shape *shape = instruction->shape();
    switch (instruction->opcode()) {
      case kAbs:
        return CreateBinary(kDivide, instruction, instruction->operand(0));
      case kAcosh:
        return CreateBinary(
            kPower,
            CreateBinary(kSubtract,
                         CreateBinary(kPower, instruction->operand(0),
                                      CreateConstant(shape, 2)),
                         CreateConstant(shape, 1)),
            CreateConstant(shape, -1 / 2.0));
      case kAsin:
        return CreateBinary(
            kPower,
            CreateBinary(kSubtract, CreateConstant(shape, 1),
                         CreateBinary(kPower, instruction->operand(0),
                                      CreateConstant(shape, 2))),
            CreateConstant(shape, -1 / 2.0));
      case kAsinh:
        return CreateBinary(
            kPower,
            CreateBinary(kAdd,
                         CreateBinary(kPower, instruction->operand(0),
                                      CreateConstant(shape, 2)),
                         CreateConstant(shape, 1)),
            CreateConstant(shape, -1 / 2.0));
      case kAtan:
        return CreateBinary(
            kDivide, CreateConstant(shape, 1),
            CreateBinary(kAdd,
                         CreateBinary(kPower, instruction->operand(0),
                                      CreateConstant(shape, 2)),
                         CreateConstant(shape, 1)));
      case kAtanh:
        return CreateBinary(
            kDivide, CreateConstant(shape, 1),
            CreateBinary(kSubtract, CreateConstant(shape, 1),
                         CreateBinary(kPower, instruction->operand(0),
                                      CreateConstant(shape, 2))));
      case kCbrt:
        return CreateBinary(kMultiply, CreateConstant(shape, 1 / 3.0),
                            CreateBinary(kPower, instruction->operand(0),
                                         CreateConstant(shape, -2 / 3.0)));
      case kCos:
        return CreateUnary(kNegate, CreateUnary(kSin, instruction->operand(0)));
      case kCosh:
        return CreateUnary(kSinh, instruction->operand(0));
      case kErf:
        return CreateBinary(
            kMultiply, CreateConstant(shape, 1.1283791671),
            CreateUnary(
                kExp, CreateUnary(kNegate,
                                  CreateBinary(kPower, instruction->operand(0),
                                               CreateConstant(shape, 2)))));
      case kExp:
        return CreateUnary(kExp, instruction->operand(0));
      case kSin:
        return CreateUnary(kCos, instruction->operand(0));
      case kSinh:
        return CreateUnary(kCosh, instruction->operand(0));
      case kSqrt:
        return CreateBinary(kMultiply, CreateConstant(shape, 1 / 2.0),
                            CreateBinary(kPower, instruction->operand(0),
                                         CreateConstant(shape, -1 / 2.0)));
      case kTan:
        return CreateBinary(kPower, CreateUnary(kCos, instruction->operand(0)),
                            CreateConstant(shape, -2));
      case kTanh:
        return CreateBinary(kPower, CreateUnary(kCosh, instruction->operand(0)),
                            CreateConstant(shape, -2));
      default:
        return nullptr;
    }
  }

 public:
  Instruction *Derivative(Instruction *instruction) {
    switch (instruction->opcode()) {
      case kAbs:
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
      case kSin:
      case kSinh:
      case kSqrt:
      case kTan:
      case kTanh:
        // Apply the chain rule
        return CreateBinary(kMultiply, ChainRuleHelper(instruction),
                            Derivative(instruction->operand(0)));
      case kAdd:
        return CreateBinary(kAdd, Derivative(instruction->operand(0)),
                            Derivative(instruction->operand(1)));
      case kConstant:
        return CreateConstant(instruction->shape(), 0);
      case kNegate:
        return CreateUnary(kNegate, Derivative(instruction->operand(0)));
      case kParameter:
        return CreateConstant(instruction->shape(), 1);
      case kSubtract:
        return CreateBinary(kSubtract, Derivative(instruction->operand(0)),
                            Derivative(instruction->operand(1)));
      default:
        return nullptr;
    }
  }
};
