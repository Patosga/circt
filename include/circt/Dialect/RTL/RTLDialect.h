//===- RTLDialect.h - RTL dialect declaration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an RTL MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLDIALECT_H
#define CIRCT_DIALECT_RTL_RTLDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace rtl {

} // namespace rtl
} // namespace circt

// Pull in the dialect definition.
#include "circt/Dialect/RTL/RTLDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/RTL/RTLEnums.h.inc"

#endif // CIRCT_DIALECT_RTL_RTLDIALECT_H
