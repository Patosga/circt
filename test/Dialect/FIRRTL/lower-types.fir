; RUN: circt-translate -import-firrtl %s | circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types)' --print-ir-before-all | FileCheck %s

circuit TypeCanonicalizationConsideredHarmful:
  module TypeCanonicalizationConsideredHarmful:

    wire a:  {a: { a: UInt<1>}}
    wire ax: {a: { a: UInt<1>}}

    a <- ax
    a.a <- ax.a
    a.a.a <- ax.a.a
    ; CHECK:      firrtl.partialconnect %a_a_a, %ax_a_a
    ; CHECK-NEXT: firrtl.partialconnect %a_a_a, %ax_a_a
    ; CHECK-NEXT: firrtl.partialconnect %a_a_a, %ax_a_a

    wire b:  { a: {flip a: UInt<1>}}
    wire bx: { a: {flip a: UInt<1>}}

    b <- bx
    b.a <- bx.a
    b.a.a <- bx.a.a
    ; CHECK:      firrtl.partialconnect %bx_a_a, %b_a_a
    ; CHECK-NEXT: firrtl.partialconnect %bx_a_a, %b_a_a
    ; CHECK-NEXT: firrtl.partialconnect %b_a_a,  %bx_a_a

    wire c:  {flip a: { a: UInt<1>}}
    wire cx: {flip a: { a: UInt<1>}}

    c <- cx
    c.a <- cx.a
    c.a.a <- cx.a.a
    ; CHECK:      firrtl.partialconnect %cx_a_a, %c_a_a
    ; CHECK-NEXT: firrtl.partialconnect %c_a_a,  %cx_a_a
    ; CHECK-NEXT: firrtl.partialconnect %c_a_a,  %cx_a_a

    wire d:  {flip a: {flip a: UInt<1>}}
    wire dx: {flip a: {flip a: UInt<1>}}

    d <- dx
    d.a <- dx.a
    d.a.a <- dx.a.a
    ; CHECK:      firrtl.partialconnect %d_a_a,  %dx_a_a
    ; CHECK-NEXT: firrtl.partialconnect %dx_a_a, %d_a_a
    ; CHECK-NEXT: firrtl.partialconnect %d_a_a,  %dx_a_a
