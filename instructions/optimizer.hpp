#include <cassert>
#include <functional>
#include <optional>

#include "evaluator.hpp"

class Optimizer {
 public:
  void Run(Instruction *instruction) {
    bool changed;
    do {
      changed = false;

      // Rewrites involving binary ops with both operands the same; this should
      // happen BEFORE recursing since there is no point in traversing the whole
      // tree if the root operation is x-x
      if (Arity(instruction->opcode()) == 2 &&
          instruction->operand(0) == instruction->operand(1)) {
        switch (instruction->opcode()) {
          case kAdd:
            VLOG(10) << "x+x --> 2*x";
            changed |= ReplaceInstruction(
                instruction,
                CreateBinary(kMultiply, instruction->operand(0),
                             CreateConstant(instruction->shape(), 2)));
            break;
          case kDivide:
            VLOG(10) << "x/x --> 1";
            changed |= ReplaceInstruction(
                instruction, CreateConstant(instruction->shape(), 1));
            break;
          case kSubtract:
            VLOG(10) << "x-x --> 0";
            changed |= ReplaceInstruction(
                instruction, CreateConstant(instruction->shape(), 0));
            break;
          default:
            break;
        }
      }

      // TODO Other pruning optimizations combine (without x+x->2x)

      // TODO Put above into HandlePruning

      // Recurse on the instruction's operands before optimizing this one
      for (auto *operand : instruction->operands()) {
        Run(operand);
      }

      // Try for arity-specific rewrites
      const auto handle_arity = GetHandleArity(Arity(instruction->opcode()));
      if (handle_arity) {
        changed |= handle_arity(instruction);
      }

      // Try for opcode-specific rewrites
      const auto handle_opcode = GetHandleOpcode(instruction->opcode());
      if (handle_opcode) {
        changed |= handle_opcode(instruction);
      }
      // Loop until we reach a fixed point
    } while (changed);
  }

 private:
  static std::function<bool(Instruction *)> GetHandleArity(int arity) {
    switch (arity) {
      case 1:
        return HandleUnary;
      case 2:
        return HandleBinary;
      default:
        return nullptr;
    }
  }

  static std::function<bool(Instruction *)> GetHandleOpcode(Opcode opcode) {
    switch (opcode) {
      case kAdd:
        return HandleAdd;
      case kDivide:
        return HandleDivide;
      case kMaximum:
        return HandleMaximum;
      case kMinimum:
        return HandleMinimum;
      case kMultiply:
        return HandleMultiply;
      case kNegate:
        return HandleNegate;
      case kPower:
        return HandlePower;
      case kSubtract:
        return HandleSubtract;
      default:
        return nullptr;
    }
  }

  static bool HandleUnary(Instruction *unary) {
    assert(Arity(unary->opcode()) == 1);

    Instruction *operand = unary->operand(0);

    // TODO Unify with analogous rewrite in HandleBinary
    if (AllConstantOperands(unary)) {
      VLOG(10) << "Folding unary operation with constant operand";
      return ReplaceInstruction(
          unary, CreateConstant(unary->shape(), Evaluate(unary)));
    }

    if (auto inv_op = Inverse(unary->opcode()); inv_op.has_value()) {
      if (operand->opcode() == Inverse(unary->opcode()).value()) {
        VLOG(10) << "f(g(x)) --> x where f and g are inverses";
        return ReplaceInstruction(unary, operand->operand(0));
      }
    }

    // sqrt(x^2) --> abs(x)
    if (unary->opcode() == kSqrt && operand->opcode() == kPower &&
        IsConstantWithValue(operand->operand(1), 2)) {
      VLOG(10) << "sqrt(x^2) --> abs(x)";
      return ReplaceInstruction(unary, CreateUnary(kAbs, operand->operand(0)));
    }

    // TODO More general: f(g(x)) --> f(x)
    if (operand->opcode() == kNegate && IsEven(unary->opcode())) {
      VLOG(10) << "f(-x) --> f(x) where f is even";
      return ReplaceInstruction(
          unary, CreateUnary(unary->opcode(), operand->operand(0)));
    }

    // TODO More general: f(g(x)) --> g(x)
    if (unary->opcode() == kAbs && IsNonNegative(operand->opcode())) {
      VLOG(10) << "|f(x)| --> f(x) where f is non-negative";
      return ReplaceInstruction(unary, operand);
    }

    // TODO More general: f(g(x)) --> g(f(x))
    if (operand->opcode() == kNegate && IsOdd(unary->opcode())) {
      VLOG(10) << "f(-x) --> -f(x) where f is odd";
      return ReplaceInstruction(
          unary, CreateUnary(kNegate, CreateUnary(unary->opcode(),
                                                  operand->operand(0))));
    }

    if (unary->opcode() == kAcosh && operand->opcode() == kCosh) {
      VLOG(10) << "acosh(cosh(x)) --> |x|";
      return ReplaceInstruction(unary, CreateUnary(kAbs, operand->operand(0)));
    }
    return false;
  }

  static bool HandleBinary(Instruction *binary) {
    assert(Arity(binary->opcode()) == 2);

    Instruction *lhs = binary->operand(0);
    Instruction *rhs = binary->operand(1);

    // Fold operations whose every operand is constant
    if (AllConstantOperands(binary)) {
      VLOG(10) << "Folding binary operation with all constant operands";
      return ReplaceInstruction(
          binary, CreateConstant(binary->shape(), Evaluate(binary)));
    }

    // Canonicalize constants to be on the lhs for commutative ops
    if (IsCommutative(binary->opcode()) && rhs->opcode() == kConstant) {
      VLOG(10) << "Canonicalizing constant to lhs of commutative binary op";
      return ReplaceInstruction(
          binary, CreateBinary(binary->opcode(), rhs, lhs));  // NOLINT
    }

    // TODO Fold expressions equal to 0: 0*x, pow(0,x)

    // TODO Fold expressions equal to 1: pow(1,x), pow(x,0), pythag

    // TODO Fold identity elements: x-0, x+0, 1*x, pow(x,1)

    // Homomorphism-style rewrites of the form f(g(x),g(y)) --> g(h(x,y))
    if (lhs->opcode() == rhs->opcode()) {
      std::optional<Opcode> new_op;
      if (binary->opcode() == kAdd && lhs->opcode() == kLog) {
        VLOG(10) << "log(x)+log(y) --> log(x*y)";
        new_op = kMultiply;
      } else if (binary->opcode() == kDivide && lhs->opcode() == kAbs) {
        VLOG(10) << "|x|/|y| --> |x/y|";
        new_op = kDivide;
      } else if (binary->opcode() == kDivide && lhs->opcode() == kExp) {
        VLOG(10) << "exp(x)/exp(y) --> exp(x-y)";
        new_op = kSubtract;
      } else if (binary->opcode() == kMultiply && lhs->opcode() == kAbs) {
        VLOG(10) << "|x|*|y| --> |x*y|";
        new_op = kMultiply;
      } else if (binary->opcode() == kMultiply && lhs->opcode() == kExp) {
        VLOG(10) << "exp(x)*exp(y) --> exp(x+y)";
        new_op = kAdd;
      } else if (binary->opcode() == kSubtract && lhs->opcode() == kLog) {
        VLOG(10) << "log(x)-log(y) --> log(x/y)";
        new_op = kDivide;
      }

      if (new_op.has_value()) {
        return ReplaceInstruction(
            binary, CreateUnary(lhs->opcode(),
                                CreateBinary(new_op.value(), lhs->operand(0),
                                             rhs->operand(0))));
      }
    }

    // TODO Strength reduction
    return false;
  }

  static bool HandleAdd(Instruction *add) {
    assert(add->opcode() == kAdd);

    Instruction *lhs = add->operand(0);
    Instruction *rhs = add->operand(1);

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "x+0 --> x";
      return ReplaceInstruction(add, rhs);
    }

    // TODO Maybe canonicalize so we don't need both of these?
    if (rhs->opcode() == kNegate) {
      VLOG(10) << "x+(-y) --> x-y";
      return ReplaceInstruction(add,
                                CreateBinary(kSubtract, lhs, rhs->operand(0)));
    }

    if (lhs->opcode() == kNegate) {
      VLOG(10) << "(-x)+y --> y-x";
      return ReplaceInstruction(add,
                                CreateBinary(kSubtract, rhs, lhs->operand(0)));
    }

    // TODO Unify these and make them more general, i.e. xy+zx --> x(y+z)
    if (lhs->opcode() == kMultiply && rhs->opcode() == kMultiply) {
      if (lhs->operand(0) == rhs->operand(0)) {
        VLOG(10) << "x*y+x*z --> x*(y+z)";
        return ReplaceInstruction(
            add,
            CreateBinary(kMultiply, lhs->operand(0),
                         CreateBinary(kAdd, lhs->operand(1), rhs->operand(1))));
      }
      if (lhs->operand(1) == rhs->operand(1)) {
        VLOG(10) << "x*z+y*z --> (x+y)*z";
        return ReplaceInstruction(
            add,
            CreateBinary(kMultiply,
                         CreateBinary(kAdd, lhs->operand(0), rhs->operand(0)),
                         lhs->operand(1)));
      }
    }

    // TODO pow(sin(x),2)+pow(cos(x),2) --> 1
    return false;
  }

  static bool HandleDivide(Instruction *divide) {
    assert(divide->opcode() == kDivide);

    Instruction *lhs = divide->operand(0);
    Instruction *rhs = divide->operand(1);

    if (rhs->opcode() == kConstant) {
      VLOG(10) << "x/c --> x*c";
      return ReplaceInstruction(
          divide,
          CreateBinary(kMultiply, lhs,
                       CreateConstant(lhs->shape(), 1 / Evaluate(rhs))));
    }

    if (lhs->opcode() == kSin && rhs->opcode() == kCos &&
        lhs->operand(0) == rhs->operand(0)) {
      VLOG(10) << "sin(x)/cos(x) --> tan(x)";
      return ReplaceInstruction(divide, CreateUnary(kTan, lhs->operand(0)));
    }

    if (rhs->opcode() == kPower) {
      VLOG(10) << "x/pow(y,z) --> x*pow(y,-z)";
      return ReplaceInstruction(
          divide,
          CreateBinary(kMultiply, lhs,
                       CreateBinary(kPower, rhs->operand(0),
                                    CreateUnary(kNegate, rhs->operand(1)))));
    }

    if (lhs->opcode() == kDivide) {
      VLOG(10) << "(x/y)/z --> x/(y*z)";
      return ReplaceInstruction(
          divide, CreateBinary(kDivide, lhs->operand(0),
                               CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }

    if (rhs->opcode() == kDivide) {
      VLOG(10) << "x/(y/z) --> (x*z)/y";
      return ReplaceInstruction(
          divide,
          CreateBinary(kDivide, CreateBinary(kMultiply, lhs, rhs->operand(1)),
                       rhs->operand(0)));
    }
    return false;
  }

  static bool HandleMaximum(Instruction *maximum) {
    assert(maximum->opcode() == kMaximum);

    Instruction *lhs = maximum->operand(0);
    Instruction *rhs = maximum->operand(1);

    // TODO Unify monotone rewrites (with sublinear condition? instance of
    // homomorphism?)
    if (lhs->opcode() == rhs->opcode() && IsIncreasing(lhs->opcode())) {
      VLOG(10) << "max(f(x), f(y)) --> f(max(x, y)) where f is increasing";
      return ReplaceInstruction(
          maximum,
          CreateUnary(lhs->opcode(), CreateBinary(kMaximum, lhs->operand(0),
                                                  rhs->operand(0))));
    }

    if (lhs->opcode() == rhs->opcode() && IsDecreasing(lhs->opcode())) {
      VLOG(10) << "max(f(x), f(y)) --> f(min(x, y)) where f is decreasing";
      return ReplaceInstruction(
          maximum,
          CreateUnary(lhs->opcode(), CreateBinary(kMinimum, lhs->operand(0),
                                                  rhs->operand(0))));
    }
    return false;
  }

  static bool HandleMinimum(Instruction *minimum) {
    assert(minimum->opcode() == kMinimum);

    Instruction *lhs = minimum->operand(0);
    Instruction *rhs = minimum->operand(1);

    if (lhs->opcode() == rhs->opcode() && IsIncreasing(lhs->opcode())) {
      VLOG(10) << "min(f(x), f(y)) --> f(min(x, y)) where f is increasing";
      return ReplaceInstruction(
          minimum,
          CreateUnary(lhs->opcode(), CreateBinary(kMinimum, lhs->operand(0),
                                                  rhs->operand(0))));
    }

    if (lhs->opcode() == rhs->opcode() && IsDecreasing(lhs->opcode())) {
      VLOG(10) << "min(f(x), f(y)) --> f(max(x, y)) where f is decreasing";
      return ReplaceInstruction(
          minimum,
          CreateUnary(lhs->opcode(), CreateBinary(kMaximum, lhs->operand(0),
                                                  rhs->operand(0))));
    }
    return false;
  }

  static bool HandleMultiply(Instruction *multiply) {
    assert(multiply->opcode() == kMultiply);

    Instruction *lhs = multiply->operand(0);
    Instruction *rhs = multiply->operand(1);

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "0*x --> 0";
      return ReplaceInstruction(multiply, lhs);
    }

    // TODO Unify with 0+x --> x add identity rewrite
    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "1*x --> x";
      return ReplaceInstruction(multiply, rhs);
    }

    // x * pow(x, y) --> pow(x, y+1)
    if (rhs->opcode() == kPower && rhs->operand(0) == lhs) {
      VLOG(10) << "x*pow(x,y) --> pow(x,y+1)";
      return ReplaceInstruction(
          multiply,
          CreateBinary(kPower, lhs,
                       CreateBinary(kAdd, rhs->operand(1),
                                    CreateConstant(lhs->shape(), 1))));
    }
    // pow(x, y) * x --> pow(x, y+1)
    if (lhs->opcode() == kPower && lhs->operand(0) == rhs) {
      VLOG(10) << "pow(x,y)*x --> pow(x,y+1)";
      return ReplaceInstruction(
          multiply,
          CreateBinary(kPower, rhs,
                       CreateBinary(kAdd, lhs->operand(1),
                                    CreateConstant(rhs->shape(), 1))));
    }

    if (lhs->opcode() == kPower && rhs->opcode() == kPower) {
      if (lhs->operand(1) == rhs->operand(1)) {
        VLOG(10) << "pow(x,z)*pow(y,z) --> pow(x*y,z)";
        return ReplaceInstruction(
            multiply, CreateBinary(kPower,
                                   CreateBinary(kMultiply, lhs->operand(0),
                                                rhs->operand(0)),
                                   lhs->operand(1)));
      }
      if (lhs->operand(0) == rhs->operand(0)) {
        VLOG(10) << "pow(x,y)*pow(x,z) --> pow(x,y+z)";
        return ReplaceInstruction(
            multiply,
            CreateBinary(kPower, lhs->operand(0),
                         CreateBinary(kAdd, lhs->operand(1), rhs->operand(1))));
      }
    }

    // (-x)*(-y) --> x*y
    if (lhs->opcode() == kNegate && rhs->opcode() == kNegate) {
      VLOG(10) << "(-x)*(-y) --> x*y";
      return ReplaceInstruction(
          multiply, CreateBinary(kMultiply, lhs->operand(0), rhs->operand(0)));
    }

    // x*(y/z) --> (x*y)/z
    if (rhs->opcode() == kDivide) {
      VLOG(10) << "x*(y/z) --> (x*y)/z";
      return ReplaceInstruction(
          multiply,
          CreateBinary(kDivide, CreateBinary(kMultiply, lhs, rhs->operand(0)),
                       rhs->operand(1)));
    }
    // (x/y)*z --> (x*z)/y
    if (lhs->opcode() == kDivide) {
      VLOG(10) << "(x/y)*z --> (x*z)/y";
      return ReplaceInstruction(
          multiply,
          CreateBinary(kDivide, CreateBinary(kMultiply, lhs->operand(0), rhs),
                       lhs->operand(1)));
    }

    return false;
  }

  static bool HandleNegate(Instruction *negate) {
    assert(negate->opcode() == kNegate);

    Instruction *operand = negate->operand(0);

    if (operand->opcode() == kSubtract) {
      VLOG(10) << "-(x-y) --> y-x";
      return ReplaceInstruction(
          negate,
          CreateBinary(kSubtract, operand->operand(1), operand->operand(0)));
    }
    return false;
  }

  static bool HandlePower(Instruction *power) {
    assert(power->opcode() == kPower);

    Instruction *lhs = power->operand(0);
    Instruction *rhs = power->operand(1);

    if (IsConstantWithValue(rhs, 0)) {
      VLOG(10) << "pow(x,0) --> 1";
      return ReplaceInstruction(power, CreateConstant(power->shape(), 1));
    }

    if (IsConstantWithValue(rhs, 1)) {
      VLOG(10) << "pow(x,1) --> x";
      return ReplaceInstruction(power, lhs);
    }

    if (IsConstantWithValue(lhs, 0)) {
      VLOG(10) << "pow(0,x) --> 0";
      return ReplaceInstruction(power, lhs);
    }

    if (IsConstantWithValue(lhs, 1)) {
      VLOG(10) << "pow(1,x) --> 1";
      return ReplaceInstruction(power, lhs);
    }

    if (lhs->opcode() == kPower) {
      VLOG(10) << "pow(pow(x,y),z) --> pow(x,y*z)";
      return ReplaceInstruction(
          power, CreateBinary(kPower, lhs->operand(0),
                              CreateBinary(kMultiply, lhs->operand(1), rhs)));
    }
    return false;
  }

  static bool HandleSubtract(Instruction *subtract) {
    assert(subtract->opcode() == kSubtract);

    // TODO pow(x,2)-pow(y,2) --> (x-y)*(x+y)
    return false;
  }
};
