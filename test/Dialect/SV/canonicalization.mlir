// RUN: circt-opt -simple-canonicalizer %s | FileCheck %s

// CHECK-LABEL: func @if_dead_condition(%arg0: i1) {
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite "Reachable1"
// CHECK-NEXT:      sv.fwrite "Reachable2"
// CHECK-NEXT:      sv.fwrite "Reachable3"
// CHECK-NEXT:      sv.fwrite "Reachable4"
// CHECK-NEXT:    }
// CHECK-NEXT:    return 
// CHECK-NEXT:  }

func @if_dead_condition(%arg0: i1) {
  sv.always posedge %arg0 {
    %true = rtl.constant true
    %false = rtl.constant false

    sv.if %true {}

    sv.if %false {
        sv.fwrite "Unreachable0"
    }

    sv.if %true {
      sv.fwrite "Reachable1"
    }

    sv.if %true {
      sv.fwrite "Reachable2"
    } else {
      sv.fwrite "Unreachable2"
    } 

    sv.if %false {
      sv.fwrite "Unreachable3"
    } else {
      sv.fwrite "Reachable3"
    } 

    sv.if %false {
      sv.fwrite "Unreachable4"
    } else {
      sv.fwrite "Reachable4"
    } 
  }

  return
}

// CHECK-LABEL: func @empy_op(%arg0: i1) {
// CHECK-NOT:     sv.if
// CHECK-NOT:     sv.ifdef
// CHECK-NOT:     sv.always
// CHECK-NOT:     sv.initial
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func @empy_op(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {}
    sv.if %arg0 {} else {}
  }
  sv.ifdef "SYNTHESIS" {}
  sv.ifdef "SYNTHESIS" {} else {}
  sv.always posedge %arg0 {}
  sv.initial {}
  return
}

// CHECK-LABEL: func @invert_if(%arg0: i1) {
// CHECK-NEXT:    %true = rtl.constant true
// CHECK-NEXT:    sv.initial  {
// CHECK-NEXT:      %0 = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.if %0  {
// CHECK-NEXT:        sv.fwrite "Foo"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func @invert_if(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {
    } else {
      sv.fwrite "Foo"
    }
  }
  return
}
