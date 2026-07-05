#include "optimizer.hpp"

#include <cmath>

#include "gtest/gtest.h"

static const Shape* one_by_one = new Shape({1, 1});

TEST(OptimizerTest, UnaryConstantFolding) {
  Instruction* c = CreateConstant(one_by_one, 5);

  Instruction* abs = CreateUnary(kAbs, c);
  Instruction* cos = CreateUnary(kCos, c);
  Instruction* exp = CreateUnary(kExp, c);
  Instruction* log = CreateUnary(kLog, c);
  Instruction* neg = CreateUnary(kNegate, c);
  Instruction* sin = CreateUnary(kSin, c);
  Instruction* tan = CreateUnary(kTan, c);
  Instruction* tanh = CreateUnary(kTanh, c);

  Optimizer opt;
  opt.Run(abs);
  EXPECT_TRUE(IsConstantWithValue(abs, 5));
  opt.Run(cos);
  EXPECT_TRUE(IsConstantWithValue(cos, std::cos(5)));
  opt.Run(exp);
  EXPECT_TRUE(IsConstantWithValue(exp, std::exp(5)));
  opt.Run(log);
  EXPECT_TRUE(IsConstantWithValue(log, std::log(5)));
  opt.Run(neg);
  EXPECT_TRUE(IsConstantWithValue(neg, -5));
  opt.Run(sin);
  EXPECT_TRUE(IsConstantWithValue(sin, std::sin(5)));
  opt.Run(tan);
  EXPECT_TRUE(IsConstantWithValue(tan, std::tan(5)));
  opt.Run(tanh);
  EXPECT_TRUE(IsConstantWithValue(tanh, std::tanh(5)));
}

TEST(OptimizerTest, BinaryConstantFolding) {
  Instruction* c1 = CreateConstant(one_by_one, 2);
  Instruction* c2 = CreateConstant(one_by_one, 3);

  Instruction* add = CreateBinary(kAdd, c1, c2);
  Instruction* div = CreateBinary(kDivide, c1, c2);
  Instruction* max = CreateBinary(kMaximum, c1, c2);
  Instruction* min = CreateBinary(kMinimum, c1, c2);
  Instruction* mul = CreateBinary(kMultiply, c1, c2);
  Instruction* pow = CreateBinary(kPower, c1, c2);
  Instruction* sub = CreateBinary(kSubtract, c1, c2);

  Optimizer opt;
  opt.Run(add);
  EXPECT_TRUE(IsConstantWithValue(add, 5));
  opt.Run(div);
  EXPECT_TRUE(IsConstantWithValue(div, 2 / 3.0));
  opt.Run(max);
  EXPECT_TRUE(IsConstantWithValue(max, 3));
  opt.Run(min);
  EXPECT_TRUE(IsConstantWithValue(min, 2));
  opt.Run(mul);
  EXPECT_TRUE(IsConstantWithValue(mul, 6));
  opt.Run(pow);
  EXPECT_TRUE(IsConstantWithValue(pow, 8));
  opt.Run(sub);
  EXPECT_TRUE(IsConstantWithValue(sub, -1));
}

TEST(OptimizerTest, BinaryCanonicalization) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* c = CreateConstant(one_by_one, 3);

  Instruction* add = CreateBinary(kAdd, x, c);
  Instruction* mul = CreateBinary(kMultiply, x, c);

  Optimizer opt;
  opt.Run(add);
  EXPECT_TRUE(add->operand(0)->opcode() == kConstant &&
              add->operand(1)->opcode() == kParameter);
  opt.Run(mul);
  EXPECT_TRUE(mul->operand(0)->opcode() == kConstant &&
              mul->operand(1)->opcode() == kParameter);
}

TEST(OptimizerTest, AddZeroToParameter) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* c = CreateConstant(one_by_one, 0);

  Instruction* add = CreateBinary(kAdd, c, x);

  Optimizer opt;
  opt.Run(add);
  EXPECT_TRUE(add->opcode() == kParameter);
}

TEST(OptimizerTest, MultiplyZeroToConstantZero) {
  Instruction* c = CreateConstant(one_by_one, 0);
  Instruction* x = CreateParameter(one_by_one);

  Instruction* mul = CreateBinary(kMultiply, c, x);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(IsConstantWithValue(mul, 0));
}

TEST(OptimizerTest, MultiplyOneToParameter) {
  Instruction* c = CreateConstant(one_by_one, 1);
  Instruction* x = CreateParameter(one_by_one);

  Instruction* mul = CreateBinary(kMultiply, c, x);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(mul->opcode() == kParameter);
}

TEST(OptimizerTest, NegateEvenFunction) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* neg = CreateUnary(kNegate, x);
  Instruction* abs = CreateUnary(kAbs, neg);
  Instruction* cos = CreateUnary(kCos, neg);

  Optimizer opt;
  opt.Run(abs);
  EXPECT_TRUE(abs->opcode() == kAbs && abs->operand(0)->opcode() == kParameter);
  opt.Run(cos);
  EXPECT_TRUE(cos->opcode() == kCos && cos->operand(0)->opcode() == kParameter);
}

TEST(OptimizerTest, NegateOddFunction) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* neg = CreateUnary(kNegate, x);
  Instruction* sin = CreateUnary(kSin, neg);
  Instruction* tan = CreateUnary(kTan, neg);
  Instruction* tanh = CreateUnary(kTanh, neg);

  Optimizer opt;
  opt.Run(sin);
  EXPECT_TRUE(sin->opcode() == kNegate && sin->operand(0)->opcode() == kSin);
  opt.Run(tan);
  EXPECT_TRUE(tan->opcode() == kNegate && tan->operand(0)->opcode() == kTan);
  opt.Run(tanh);
  EXPECT_TRUE(tanh->opcode() == kNegate && tanh->operand(0)->opcode() == kTanh);
}

TEST(OptimizerTest, InverseCancellation) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* exp = CreateUnary(kExp, x);
  Instruction* log = CreateUnary(kLog, x);
  Instruction* neg = CreateUnary(kNegate, x);

  Instruction* log_exp = CreateUnary(kLog, exp);
  Instruction* exp_log = CreateUnary(kExp, log);
  Instruction* neg_neg = CreateUnary(kNegate, neg);

  Optimizer opt;
  opt.Run(log_exp);
  EXPECT_TRUE(log_exp->opcode() == kParameter);
  opt.Run(exp_log);
  EXPECT_TRUE(exp_log->opcode() == kParameter);
  opt.Run(neg_neg);
  EXPECT_TRUE(neg_neg->opcode() == kParameter);
}

TEST(OptimizerTest, MultiplyExponentials) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* exp_x = CreateUnary(kExp, x);
  Instruction* neg = CreateUnary(kNegate, x);
  Instruction* exp_neg = CreateUnary(kExp, neg);
  Instruction* mul = CreateBinary(kMultiply, exp_x, exp_neg);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(IsConstantWithValue(mul, 1));
}

TEST(OptimizerTest, MultiplyPowers) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* two = CreateConstant(one_by_one, 2);
  Instruction* pow1 = CreateBinary(kPower, x, two);
  Instruction* four = CreateConstant(one_by_one, 4);
  Instruction* pow2 = CreateBinary(kPower, x, four);
  Instruction* mul = CreateBinary(kMultiply, pow1, pow2);

  Optimizer opt;
  opt.Run(mul);
  // TODO This seems to be flaky?
  EXPECT_TRUE(mul->opcode() == kPower &&
              IsConstantWithValue(mul->operand(1), 6));
}

TEST(OptimizerTest, DividePowers) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* three = CreateConstant(one_by_one, 3);
  Instruction* pow = CreateBinary(kPower, x, three);
  Instruction* one = CreateConstant(one_by_one, 1);
  Instruction* div = CreateBinary(kDivide, one, pow);

  Optimizer opt;
  opt.Run(div);
  EXPECT_TRUE(div->opcode() == kPower &&
              IsConstantWithValue(div->operand(1), -3));
}

TEST(OptimizerTest, DivideByConstantToMultiply) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* two = CreateConstant(one_by_one, 2);
  Instruction* div = CreateBinary(kDivide, x, two);

  Optimizer opt;
  opt.Run(div);
  EXPECT_TRUE(div->opcode() == kMultiply &&
              IsConstantWithValue(div->operand(0), 0.5));
}

TEST(OptimizerTest, SineOverCosineToTangent) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* sin = CreateUnary(kSin, x);
  Instruction* cos = CreateUnary(kCos, x);
  Instruction* div = CreateBinary(kDivide, sin, cos);

  Optimizer opt;
  opt.Run(div);
  EXPECT_TRUE(div->opcode() == kTan);
  EXPECT_TRUE(div->operand(0) == x);
}

TEST(OptimizerTest, CancelAbsExponential) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* exp = CreateUnary(kExp, x);
  Instruction* abs = CreateUnary(kAbs, exp);

  Optimizer opt;
  opt.Run(abs);
  EXPECT_TRUE(abs->opcode() == kExp);
  EXPECT_TRUE(abs->operand(0) == x);
}

TEST(OptimizerTest, FactorAddToMultiply) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);
  Instruction* z = CreateParameter(one_by_one);

  Instruction* lhs = CreateBinary(kMultiply, x, y);
  Instruction* rhs = CreateBinary(kMultiply, x, z);
  Instruction* add = CreateBinary(kAdd, lhs, rhs);

  Optimizer opt;
  opt.Run(add);
  EXPECT_TRUE(add->opcode() == kMultiply && add->operand(1)->opcode() == kAdd);
}

TEST(OptimizerTest, AddWithSameOperands) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* add = CreateBinary(kAdd, x, x);

  Optimizer opt;
  opt.Run(add);
  EXPECT_TRUE(add->opcode() == kMultiply);
}

TEST(OptimizerTest, DivideDivideToMultiply) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);
  Instruction* z = CreateParameter(one_by_one);

  Instruction* div_lhs = CreateBinary(kDivide, CreateBinary(kDivide, x, y), z);
  Instruction* div_rhs = CreateBinary(kDivide, x, CreateBinary(kDivide, y, z));

  Optimizer opt;
  opt.Run(div_lhs);
  opt.Run(div_rhs);
  EXPECT_TRUE(div_lhs->opcode() == kDivide && div_rhs->opcode() == kDivide);
  EXPECT_TRUE(div_lhs->operand(1)->opcode() == kMultiply &&
              div_rhs->operand(0)->opcode() == kMultiply);
}

// TODO This is just an instance of a homomorphism.
TEST(OptimizerTest, ApplyMonotonicHomomorphism) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);

  Instruction* exp1 = CreateUnary(kExp, x);
  Instruction* exp2 = CreateUnary(kExp, y);
  Instruction* neg1 = CreateUnary(kNegate, x);
  Instruction* neg2 = CreateUnary(kNegate, y);

  Instruction* max1 = CreateBinary(kMaximum, exp1, exp2);
  Instruction* max2 = CreateBinary(kMaximum, neg1, neg2);
  Instruction* min1 = CreateBinary(kMinimum, exp1, exp2);
  Instruction* min2 = CreateBinary(kMinimum, neg1, neg2);

  Optimizer opt;
  opt.Run(max1);
  EXPECT_TRUE(max1->opcode() == kExp && max1->operand(0)->opcode() == kMaximum);
  opt.Run(max2);
  EXPECT_TRUE(max2->opcode() == kNegate &&
              max2->operand(0)->opcode() == kMinimum);
  opt.Run(min1);
  EXPECT_TRUE(min1->opcode() == kExp && min1->operand(0)->opcode() == kMinimum);
  opt.Run(min2);
  EXPECT_TRUE(min2->opcode() == kNegate &&
              min2->operand(0)->opcode() == kMaximum);
}

TEST(OptimizerTest, MultiplyLikePowersToPower) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);
  Instruction* z = CreateParameter(one_by_one);

  Instruction* pow1 = CreateBinary(kPower, x, z);
  Instruction* pow2 = CreateBinary(kPower, y, z);
  Instruction* mul = CreateBinary(kMultiply, pow1, pow2);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(mul->opcode() == kPower &&
              mul->operand(0)->opcode() == kMultiply);
}

TEST(OptimizerTest, MultiplyLikeBaseToPower) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);
  Instruction* z = CreateParameter(one_by_one);

  Instruction* pow1 = CreateBinary(kPower, x, y);
  Instruction* pow2 = CreateBinary(kPower, x, z);
  Instruction* mul = CreateBinary(kMultiply, pow1, pow2);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(mul->opcode() == kPower && mul->operand(1)->opcode() == kAdd);
}

TEST(OptimizerTest, NegateSubtractToSubtract) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);

  Instruction* sub = CreateBinary(kSubtract, x, y);
  Instruction* neg = CreateUnary(kNegate, sub);

  Optimizer opt;
  opt.Run(neg);
  EXPECT_TRUE(neg->opcode() == kSubtract);
  EXPECT_TRUE(neg->operand(0) == y && neg->operand(1) == x);
}

TEST(OptimizerTest, PowerZeroToConstantOne) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* c = CreateConstant(one_by_one, 0);

  Instruction* pow = CreateBinary(kPower, x, c);

  Optimizer opt;
  opt.Run(pow);
  EXPECT_TRUE(IsConstantWithValue(pow, 1));
}

TEST(OptimizerTest, PowerOneToParameter) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* c = CreateConstant(one_by_one, 1);

  Instruction* pow = CreateBinary(kPower, x, c);

  Optimizer opt;
  opt.Run(pow);
  EXPECT_TRUE(pow->opcode() == kParameter);
}

TEST(OptimizerTest, ZeroPowerToConstantZero) {
  Instruction* c = CreateConstant(one_by_one, 0);
  Instruction* x = CreateParameter(one_by_one);

  Instruction* pow = CreateBinary(kPower, c, x);

  Optimizer opt;
  opt.Run(pow);
  EXPECT_TRUE(IsConstantWithValue(pow, 0));
}

TEST(OptimizerTest, OnePowerToConstantOne) {
  Instruction* c = CreateConstant(one_by_one, 1);
  Instruction* x = CreateParameter(one_by_one);

  Instruction* pow = CreateBinary(kPower, c, x);

  Optimizer opt;
  opt.Run(pow);
  EXPECT_TRUE(IsConstantWithValue(pow, 1));
}

TEST(OptimizerTest, PowerPowerToPowerMultiply) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);
  Instruction* z = CreateParameter(one_by_one);

  Instruction* pow1 = CreateBinary(kPower, x, y);
  Instruction* pow2 = CreateBinary(kPower, pow1, z);

  Optimizer opt;
  opt.Run(pow2);
  EXPECT_TRUE(pow2->opcode() == kPower);
  EXPECT_TRUE(pow2->operand(0) == x);
  EXPECT_TRUE(pow2->operand(1)->opcode() == kMultiply);
  EXPECT_TRUE(pow2->operand(1)->operand(0) == y);
  EXPECT_TRUE(pow2->operand(1)->operand(1) == z);
}

TEST(OptimizerTest, CancelTrigInverses) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* cos_acos = CreateUnary(kCos, CreateUnary(kAcos, x));
  Instruction* sin_asin = CreateUnary(kSin, CreateUnary(kAsin, x));
  Instruction* tan_atan = CreateUnary(kTan, CreateUnary(kAtan, x));

  Optimizer opt;
  opt.Run(cos_acos);
  EXPECT_TRUE(cos_acos->opcode() == kParameter);
  opt.Run(sin_asin);
  EXPECT_TRUE(sin_asin->opcode() == kParameter);
  opt.Run(tan_atan);
  EXPECT_TRUE(tan_atan->opcode() == kParameter);
}

TEST(OptimizerTest, CancelHyperbolicInverses) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* acosh_cosh = CreateUnary(kAcosh, CreateUnary(kCosh, x));
  Instruction* asinh_sinh = CreateUnary(kAsinh, CreateUnary(kSinh, x));
  Instruction* atanh_tanh = CreateUnary(kAtanh, CreateUnary(kTanh, x));
  Instruction* cosh_acosh = CreateUnary(kCosh, CreateUnary(kAcosh, x));
  Instruction* sinh_asinh = CreateUnary(kSinh, CreateUnary(kAsinh, x));
  Instruction* tanh_atanh = CreateUnary(kTanh, CreateUnary(kAtanh, x));

  Optimizer opt;
  opt.Run(acosh_cosh);
  EXPECT_TRUE(acosh_cosh->opcode() == kAbs);
  opt.Run(asinh_sinh);
  EXPECT_TRUE(asinh_sinh->opcode() == kParameter);
  opt.Run(atanh_tanh);
  EXPECT_TRUE(atanh_tanh->opcode() == kParameter);
  opt.Run(cosh_acosh);
  EXPECT_TRUE(cosh_acosh->opcode() == kParameter);
  opt.Run(sinh_asinh);
  EXPECT_TRUE(sinh_asinh->opcode() == kParameter);
  opt.Run(tanh_atanh);
  EXPECT_TRUE(tanh_atanh->opcode() == kParameter);
}

TEST(OptimizerTest, ApplyHomomorphisms) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);

  Instruction* add_logs =
      CreateBinary(kAdd, CreateUnary(kLog, x), CreateUnary(kLog, y));
  Instruction* div_abss =
      CreateBinary(kDivide, CreateUnary(kAbs, x), CreateUnary(kAbs, y));
  Instruction* div_exps =
      CreateBinary(kDivide, CreateUnary(kExp, x), CreateUnary(kExp, y));
  Instruction* mul_abss =
      CreateBinary(kMultiply, CreateUnary(kAbs, x), CreateUnary(kAbs, y));
  Instruction* mul_exps =
      CreateBinary(kMultiply, CreateUnary(kExp, x), CreateUnary(kExp, y));
  Instruction* sub_logs =
      CreateBinary(kSubtract, CreateUnary(kLog, x), CreateUnary(kLog, y));

  Optimizer opt;
  opt.Run(add_logs);
  EXPECT_TRUE(add_logs->opcode() == kLog &&
              add_logs->operand(0)->opcode() == kMultiply);
  opt.Run(div_abss);
  EXPECT_TRUE(div_abss->opcode() == kAbs &&
              div_abss->operand(0)->opcode() == kDivide);
  opt.Run(div_exps);
  EXPECT_TRUE(div_exps->opcode() == kExp &&
              div_exps->operand(0)->opcode() == kSubtract);
  opt.Run(mul_abss);
  EXPECT_TRUE(mul_abss->opcode() == kAbs &&
              mul_abss->operand(0)->opcode() == kMultiply);
  opt.Run(mul_exps);
  EXPECT_TRUE(mul_exps->opcode() == kExp &&
              mul_exps->operand(0)->opcode() == kAdd);
  opt.Run(sub_logs);
  EXPECT_TRUE(sub_logs->opcode() == kLog &&
              sub_logs->operand(0)->opcode() == kDivide);
}

TEST(OptimizerTest, MultiplyParameterPowerToPower) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* two = CreateConstant(one_by_one, 2);
  Instruction* pow = CreateBinary(kPower, x, two);
  Instruction* mul = CreateBinary(kMultiply, x, pow);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(mul->opcode() == kPower &&
              IsConstantWithValue(mul->operand(1), 3));
}

// TODO Generalize inverse cancellation to also handle this.
TEST(OptimizerTest, SquareOfSquareRootToAbs) {
  Instruction* x = CreateParameter(one_by_one);

  Instruction* two = CreateConstant(one_by_one, 2);
  Instruction* pow = CreateBinary(kPower, x, two);
  Instruction* sqrt = CreateUnary(kSqrt, pow);

  Optimizer opt;
  opt.Run(sqrt);
  EXPECT_TRUE(sqrt->opcode() == kAbs);
}

TEST(OptimizerTest, MultiplyNegativeCancellation) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);

  Instruction* neg_x = CreateUnary(kNegate, x);
  Instruction* neg_y = CreateUnary(kNegate, y);
  Instruction* mul = CreateBinary(kMultiply, neg_x, neg_y);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(mul->operand(0)->opcode() == kParameter &&
              mul->operand(1)->opcode() == kParameter);
}

TEST(OptimizerTest, MultiplyDivideToMultiply) {
  Instruction* x = CreateParameter(one_by_one);
  Instruction* y = CreateParameter(one_by_one);
  Instruction* z = CreateParameter(one_by_one);

  Instruction* div = CreateBinary(kDivide, y, z);
  Instruction* mul = CreateBinary(kMultiply, x, div);

  Optimizer opt;
  opt.Run(mul);
  EXPECT_TRUE(mul->opcode() == kDivide);
}
