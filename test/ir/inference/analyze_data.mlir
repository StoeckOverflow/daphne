// RUN: split-file %s %t
// RUN: daphne-opt %t/valid.mlir --verify-each | daphne-opt --verify-each | FileCheck %s
// RUN: daphne-opt %t/valid.mlir --canonicalize --cse --verify-each | FileCheck %s
// RUN: daphne-opt %t/invalid.mlir --split-input-file --verify-diagnostics

//--- valid.mlir
func.func @requests(%arg0: !daphne.Matrix<?x?xf64>, %arg1: i1) {
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64>, i1, i1, i1, i1) -> ()
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64>, i1, i1, i1, i1) -> ()
  return
}
// CHECK-LABEL: func.func @requests(%arg0: !daphne.Matrix<?x?xf64>, %arg1: i1) {
// CHECK-NEXT: "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64>, i1, i1, i1, i1) -> ()
// CHECK-NEXT: "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64>, i1, i1, i1, i1) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @column(%arg0: !daphne.Column<?xsi64>, %arg1: i1) {
  %0 = "daphne.constant"() <{value = false}> : () -> i1
  "daphne.analyzeData"(%arg0, %0, %0, %arg1, %0) : (!daphne.Column<?xsi64>, i1, i1, i1, i1) -> ()
  return
}
// CHECK-LABEL: func.func @column(%arg0: !daphne.Column<?xsi64>, %arg1: i1) {
// CHECK-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// CHECK-NEXT: "daphne.analyzeData"(%arg0, %0, %0, %arg1, %0) : (!daphne.Column<?xsi64>, i1, i1, i1, i1) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @integer_input(%arg0: !daphne.Matrix<?x?xsi64>, %arg1: i1) {
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xsi64>, i1, i1, i1, i1) -> ()
  return
}
// CHECK-LABEL: func.func @integer_input(%arg0: !daphne.Matrix<?x?xsi64>, %arg1: i1) {
// CHECK-NEXT: "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xsi64>, i1, i1, i1, i1) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @sparse_input(%arg0: !daphne.Matrix<?x?xf64:rep[sparse]>, %arg1: i1) {
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64:rep[sparse]>, i1, i1, i1, i1) -> ()
  return
}
// CHECK-LABEL: func.func @sparse_input(%arg0: !daphne.Matrix<?x?xf64:rep[sparse]>, %arg1: i1) {
// CHECK-NEXT: "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64:rep[sparse]>, i1, i1, i1, i1) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

//--- invalid.mlir
func.func @non_boolean_request(%arg0: !daphne.Matrix<?x?xf64>, %arg1: si64, %arg2: i1) {
  // expected-error@+1 {{operand #1 must be}}
  "daphne.analyzeData"(%arg0, %arg1, %arg2, %arg2, %arg2) : (!daphne.Matrix<?x?xf64>, si64, i1, i1, i1) -> ()
  return
}
// -----
func.func @unexpected_result(%arg0: !daphne.Matrix<?x?xf64>, %arg1: i1) {
  // expected-error@+1 {{requires zero results}}
  %0 = "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xf64>, i1, i1, i1, i1) -> ui64
  return
}
// -----
func.func @frame_input(%arg0: !daphne.Frame<?x[?: f64]>, %arg1: i1) {
  %0 = "daphne.constant"() <{value = false}> : () -> i1
  // expected-error@+1 {{requires a numeric matrix or column}}
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Frame<?x[?: f64]>, i1, i1, i1, i1) -> ()
  return
}

// -----
func.func @unsupported_column_request(%arg0: !daphne.Column<?xsi64>, %arg1: i1) {
  %0 = "daphne.constant"() <{value = false}> : () -> i1
  // expected-error@+1 {{columns support bounds acquisition only}}
  "daphne.analyzeData"(%arg0, %0, %0, %0, %arg1) : (!daphne.Column<?xsi64>, i1, i1, i1, i1) -> ()
  return
}

// -----
func.func @boolean_input(%arg0: !daphne.Matrix<?x?xi1>, %arg1: i1) {
  // expected-error@+1 {{requires a numeric matrix or column}}
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xi1>, i1, i1, i1, i1) -> ()
  return
}

// -----
func.func @unsupported_integer_width(%arg0: !daphne.Matrix<?x?xsi16>, %arg1: i1) {
  // expected-error@+1 {{requires a numeric matrix or column}}
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xsi16>, i1, i1, i1, i1) -> ()
  return
}
// -----
func.func @signless_integer(%arg0: !daphne.Matrix<?x?xi32>, %arg1: i1) {
  // expected-error@+1 {{requires a numeric matrix or column}}
  "daphne.analyzeData"(%arg0, %arg1, %arg1, %arg1, %arg1) : (!daphne.Matrix<?x?xi32>, i1, i1, i1, i1) -> ()
  return
}
