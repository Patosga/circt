//===- StandardToHandshake.cpp - Convert standard MLIR into dataflow IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This is the main Standard to Handshake Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "../PassDetail.h"
#include "circt/Conversion/StandardToHandshake/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// Visit affine expressions recursively and build the sequence of operations
/// that correspond to it.  Visitation functions return an Value of the
/// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  /// This internal class expects arguments to be non-null, checks must be
  /// performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy>
  Value buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<MulIOp>(expr);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
  Value visitModExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "modulo by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value remainder = builder.create<SignedRemIOp>(loc, lhs, rhs);
    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value isRemainderNegative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder = builder.create<AddIOp>(loc, remainder, rhs);
    Value result = builder.create<SelectOp>(loc, isRemainderNegative,
                                            correctedRemainder, remainder);
    return result;
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value noneCst = builder.create<ConstantIndexOp>(loc, -1);
    Value negative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, zeroCst);
    Value negatedDecremented = builder.create<SubIOp>(loc, noneCst, lhs);
    Value dividend =
        builder.create<SelectOp>(loc, negative, negatedDecremented, lhs);
    Value quotient = builder.create<SignedDivIOp>(loc, dividend, rhs);
    Value correctedQuotient = builder.create<SubIOp>(loc, noneCst, quotient);
    Value result =
        builder.create<SelectOp>(loc, negative, correctedQuotient, quotient);
    return result;
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(loc) << "semi-affine expressions (division by non-const) are "
                        "not supported";
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value oneCst = builder.create<ConstantIndexOp>(loc, 1);
    Value nonPositive =
        builder.create<CmpIOp>(loc, CmpIPredicate::sle, lhs, zeroCst);
    Value negated = builder.create<SubIOp>(loc, zeroCst, lhs);
    Value decremented = builder.create<SubIOp>(loc, lhs, oneCst);
    Value dividend =
        builder.create<SelectOp>(loc, nonPositive, negated, decremented);
    Value quotient = builder.create<SignedDivIOp>(loc, dividend, rhs);
    Value negatedQuotient = builder.create<SubIOp>(loc, zeroCst, quotient);
    Value incrementedQuotient = builder.create<AddIOp>(loc, quotient, oneCst);
    Value result = builder.create<SelectOp>(loc, nonPositive, negatedQuotient,
                                            incrementedQuotient);
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op =
        builder.create<ConstantOp>(loc, builder.getIndexType(), valueAttr);
    return op.getResult();
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ValueRange dimValues;
  ValueRange symbolValues;

  Location loc;
};
} // namespace

/// Create a sequence of operations that implement the `expr` applied to the
/// given dimension and symbol values.
mlir::Value mlir::expandAffineExpr(OpBuilder &builder, Location loc,
                                   AffineExpr expr, ValueRange dimValues,
                                   ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value, 8>> mlir::expandAffineMap(OpBuilder &builder,
                                                      Location loc,
                                                      AffineMap affineMap,
                                                      ValueRange operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = llvm::to_vector<8>(
      llvm::map_range(affineMap.getResults(),
                      [numDims, &builder, loc, operands](AffineExpr expr) {
                        return expandAffineExpr(builder, loc, expr,
                                                operands.take_front(numDims),
                                                operands.drop_front(numDims));
                      }));
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return None;
}

/// Rewrite affine.for operations in a handshake.func into its representations
/// as a CFG in the standard dialect. Affine expressions in loop bounds will be
/// expanded to code in the standard dialect that actually computes them. We
/// combine the lowering of affine loops in the following two conversions:
/// [AffineToStandard](https://mlir.llvm.org/doxygen/AffineToStandard_8cpp.html),
/// [SCFToStandard](https://mlir.llvm.org/doxygen/SCFToStandard_8cpp_source.html)
/// into this function.
/// The affine memory operations will be preserved until other rewrite
/// functions, e.g.,`replaceMemoryOps`, are called. Any affine analysis, e.g.,
/// getting dependence information, should be carried out before calling this
/// function; otherwise, the affine for loops will be destructed and key
/// information will be missing.
LogicalResult rewriteAffineFor(FuncOp f, ConversionPatternRewriter &rewriter) {
  // Get all affine.for operations in the function body.
  SmallVector<mlir::AffineForOp, 8> forOps;
  f.walk([&](mlir::AffineForOp op) { forOps.push_back(op); });

  // TODO: how to deal with nested loops?
  for (unsigned i = 0, e = forOps.size(); i < e; i++) {
    auto forOp = forOps[i];

    // Insert lower and upper bounds right at the position of the original
    // affine.for operation.
    rewriter.setInsertionPoint(forOp);
    auto loc = forOp.getLoc();
    auto lowerBound = expandAffineMap(rewriter, loc, forOp.getLowerBoundMap(),
                                      forOp.getLowerBoundOperands());
    auto upperBound = expandAffineMap(rewriter, loc, forOp.getUpperBoundMap(),
                                      forOp.getUpperBoundOperands());
    if (!lowerBound || !upperBound)
      return failure();
    auto step = rewriter.create<mlir::ConstantIndexOp>(loc, forOp.getStep());

    // Build blocks for a common for loop. initBlock and initPosition are the
    // block that contains the current forOp, and the position of the forOp.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();

    // Split the current block into several parts. `endBlock` contains the code
    // starting from forOp. `conditionBlock` will have the condition branch.
    // `firstBodyBlock` is the loop body, and `lastBodyBlock` is about the loop
    // iterator stepping. Here we move the body region of the AffineForOp here
    // and split it into `conditionBlock`, `firstBodyBlock`, and
    // `lastBodyBlock`.
    // TODO: is there a simpler API for doing so?
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);
    // Split and get the references to different parts (blocks) of the original
    // loop body.
    auto *conditionBlock = &forOp.region().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock =
        rewriter.splitBlock(firstBodyBlock, firstBodyBlock->end());
    rewriter.inlineRegionBefore(forOp.region(), endBlock);

    // The loop IV is the first argument of the conditionBlock.
    auto iv = conditionBlock->getArgument(0);

    // Get the loop terminator, which should be the last operation of the
    // original loop body. And `firstBodyBlock` points to that loop body.
    auto terminator = dyn_cast<mlir::AffineYieldOp>(firstBodyBlock->back());
    if (!terminator)
      return failure();

    // First, we fill the content of the lastBodyBlock with how the loop
    // iterator steps.
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto stepped = rewriter.create<mlir::AddIOp>(loc, iv, step).getResult();

    // Next, we get the loop carried values, which are terminator operands.
    SmallVector<Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator.operand_begin(), terminator.operand_end());
    rewriter.create<mlir::BranchOp>(loc, conditionBlock, loopCarried);

    // Then we fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::slt, iv,
                                                    upperBound.getValue()[0]);

    rewriter.create<mlir::CondBranchOp>(loc, comparison, firstBodyBlock,
                                        ArrayRef<Value>(), endBlock,
                                        ArrayRef<Value>());

    // We also insert the branch operation at the end of the initBlock.
    rewriter.setInsertionPointToEnd(initBlock);
    // TODO: should we add more operands here?
    rewriter.create<mlir::BranchOp>(loc, conditionBlock,
                                    lowerBound.getValue()[0]);

    // Finally, setup the firstBodyBlock.
    rewriter.setInsertionPointToEnd(firstBodyBlock);
    // TODO: is it necessary to add this explicit branch operation?
    rewriter.create<mlir::BranchOp>(loc, lastBodyBlock);

    // Remove the original forOp and the terminator in the loop body.
    rewriter.eraseOp(terminator);
    rewriter.eraseOp(forOp);
  }

  return success();
}

struct FuncOpLowering : public OpConversionPattern<mlir::FuncOp> {
  using OpConversionPattern<mlir::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Rewrite affine.for operations.
    if (failed(rewriteAffineFor(funcOp, rewriter)))
      return funcOp.emitOpError("failed to rewrite Affine loops");

    return success();
  }
};

struct AffineToStdPass : public circt::AffineToStdBase<AffineToStdPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.insert<FuncOpLowering>(m.getContext());

    llvm::errs() << "Hello world from AffineToStd!"
                 << "\n";
    llvm::errs() << m;
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::createAffineToStdPass() {
  return std::make_unique<AffineToStdPass>();
}