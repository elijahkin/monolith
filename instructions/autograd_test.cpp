#include "autograd.hpp"

#include "gtest/gtest.h"

TEST(AutogradTest, TangentDerivative) {
  const Shape* two_by_three = new Shape({2, 3});

  Instruction* x = CreateParameter(two_by_three);
  Instruction* tan = CreateUnary(kTan, x);
  VLOG(10) << tan->to_string();

  Autograd grad;
  Instruction* deriv = grad.Derivative(tan);
  VLOG(10) << deriv->to_string();
}
