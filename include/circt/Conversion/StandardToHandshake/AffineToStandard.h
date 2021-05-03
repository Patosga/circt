//===- StandardToHandshake.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Affine dialect to
// the Standard dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AFFINETOSTANDARD_H_
#define CIRCT_CONVERSION_AFFINETOSTANDARD_H_

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAffineToStdPass();
} // namespace circt
#endif