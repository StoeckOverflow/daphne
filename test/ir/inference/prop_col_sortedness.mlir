// RUN: split-file %s %t
// RUN: daphne-opt %t/adapt.mlir --adapt-types-to-kernels --verify-each | FileCheck %s --check-prefix=ADAPT
// RUN: daphne-opt %t/inference.mlir --inference --verify-each | FileCheck %s --check-prefix=INFER
// RUN: daphne-opt %t/parse.mlir --verify-each | daphne-opt --verify-each | FileCheck %s --check-prefix=PARSE
// RUN: daphne-opt %t/invalid.mlir --split-input-file --verify-diagnostics
// RUN: daphne-opt %t/transfer.mlir --transfer-data-props --verify-each | FileCheck %s --check-prefix=TRANSFER

//--- inference.mlir
// Each column is independent. A matrix/scalar less-than reverses integer order.
func.func @comparison(%arg0: !daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, %arg1: si64) {
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, si64) -> !daphne.Matrix<4x3xsi64>
  %1 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, !daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>) -> !daphne.Matrix<4x3xsi64>
  return
}
// INFER-LABEL: func.func @comparison(%arg0: !daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, %arg1: si64) {
// INFER-NEXT: %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, si64) -> !daphne.Matrix<4x3xsi64:colSortedness[descending, unknown, ascending]:lowerBound[0.000000e+00]:upperBound[1.000000e+00]>
// INFER-NEXT: %1 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, !daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>) -> !daphne.Matrix<4x3xsi64>
// INFER-NEXT: return
// INFER-NEXT: }

// Lexicographic secondary keys are not globally sorted. Dynamic directions and index results stay unknown.
func.func @ordering(%arg0: !daphne.Matrix<4x3xsi64>, %arg1: i1) {
  %0 = "daphne.constant"() <{value = 1 : index}> : () -> index
  %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %2 = "daphne.constant"() <{value = true}> : () -> i1
  %3 = "daphne.constant"() <{value = false}> : () -> i1
  %4 = "daphne.order"(%arg0, %0, %1, %2, %3, %3) : (!daphne.Matrix<4x3xsi64>, index, index, i1, i1, i1) -> !daphne.Matrix<4x3xsi64>
  %5 = "daphne.order"(%arg0, %1, %3, %3) : (!daphne.Matrix<4x3xsi64>, index, i1, i1) -> !daphne.Matrix<4x3xsi64>
  %6 = "daphne.order"(%arg0, %0, %arg1, %3) : (!daphne.Matrix<4x3xsi64>, index, i1, i1) -> !daphne.Matrix<4x3xsi64>
  %7 = "daphne.order"(%arg0, %0, %2, %2) : (!daphne.Matrix<4x3xsi64>, index, i1, i1) -> !daphne.Matrix<4x1xsi64>
  return
}
// INFER-LABEL: func.func @ordering(%arg0: !daphne.Matrix<4x3xsi64>, %arg1: i1) {
// INFER-NEXT: %0 = "daphne.constant"() <{value = 1 : index}> : () -> index
// INFER-NEXT: %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
// INFER-NEXT: %2 = "daphne.constant"() <{value = true}> : () -> i1
// INFER-NEXT: %3 = "daphne.constant"() <{value = false}> : () -> i1
// INFER-NEXT: %4 = "daphne.order"(%arg0, %0, %1, %2, %3, %3) : (!daphne.Matrix<4x3xsi64>, index, index, i1, i1, i1) -> !daphne.Matrix<4x3xsi64:colSortedness[unknown, ascending, unknown]>
// INFER-NEXT: %5 = "daphne.order"(%arg0, %1, %3, %3) : (!daphne.Matrix<4x3xsi64>, index, i1, i1) -> !daphne.Matrix<4x3xsi64:colSortedness[unknown, unknown, descending]>
// INFER-NEXT: %6 = "daphne.order"(%arg0, %0, %arg1, %3) : (!daphne.Matrix<4x3xsi64>, index, i1, i1) -> !daphne.Matrix<4x3xsi64>
// INFER-NEXT: %7 = "daphne.order"(%arg0, %0, %2, %2) : (!daphne.Matrix<4x3xsi64>, index, i1, i1) -> !daphne.Matrix<4x1xsi64:lowerBound[0.000000e+00]:upperBound[3.000000e+00]>
// INFER-NEXT: return
// INFER-NEXT: }

// Floating comparisons, arithmetic and transposition do not inherit ordering.
func.func @unsupported(%arg0: !daphne.Matrix<4x2xf64:colSortedness[ascending, descending]>, %arg1: f64) {
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<4x2xf64:colSortedness[ascending, descending]>, f64) -> !daphne.Matrix<4x2xf64>
  %1 = "daphne.transpose"(%arg0) : (!daphne.Matrix<4x2xf64:colSortedness[ascending, descending]>) -> !daphne.Matrix<2x4xf64>
  return
}
// INFER-LABEL: func.func @unsupported(%arg0: !daphne.Matrix<4x2xf64:colSortedness[ascending, descending]>, %arg1: f64) {
// INFER-NEXT: %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<4x2xf64:colSortedness[ascending, descending]>, f64) -> !daphne.Matrix<4x2xf64:lowerBound[0.000000e+00]:upperBound[1.000000e+00]>
// INFER-NEXT: %1 = "daphne.transpose"(%arg0) : (!daphne.Matrix<4x2xf64:colSortedness[ascending, descending]>) -> !daphne.Matrix<2x4xf64>
// INFER-NEXT: return
// INFER-NEXT: }

// Constant columns are nondecreasing, including empty and single-row matrices.
func.func @fill(%arg0: f64) {
  %0 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %1 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %2 = "daphne.constant"() <{value = 1 : index}> : () -> index
  %3 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
  %4 = "daphne.constant"() <{value = 0x7FF8000000000000 : f64}> : () -> f64
  %5 = "daphne.fill"(%3, %0, %0) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
  %6 = "daphne.fill"(%3, %1, %0) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
  %7 = "daphne.fill"(%3, %2, %0) : (si64, index, index) -> !daphne.Matrix<?x?xsi64>
  %8 = "daphne.fill"(%arg0, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  %9 = "daphne.fill"(%4, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  return
}
// INFER-LABEL: func.func @fill(%arg0: f64) {
// INFER-NEXT: %0 = "daphne.constant"() <{value = 2 : index}> : () -> index
// INFER-NEXT: %1 = "daphne.constant"() <{value = 0 : index}> : () -> index
// INFER-NEXT: %2 = "daphne.constant"() <{value = 1 : index}> : () -> index
// INFER-NEXT: %3 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
// INFER-NEXT: %4 = "daphne.constant"() <{value = 0x7FF8000000000000 : f64}> : () -> f64
// INFER-NEXT: %5 = "daphne.fill"(%3, %0, %0) : (si64, index, index) -> !daphne.Matrix<2x2xsi64:symmetric[true]:numDistinct[1]:colSortedness[ascending, ascending]>
// INFER-NEXT: %6 = "daphne.fill"(%3, %1, %0) : (si64, index, index) -> !daphne.Matrix<0x2xsi64:symmetric[false]:numDistinct[0]:colSortedness[ascending, ascending]>
// INFER-NEXT: %7 = "daphne.fill"(%3, %2, %0) : (si64, index, index) -> !daphne.Matrix<1x2xsi64:symmetric[false]:numDistinct[1]:colSortedness[ascending, ascending]>
// INFER-NEXT: %8 = "daphne.fill"(%arg0, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
// INFER-NEXT: %9 = "daphne.fill"(%4, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
// INFER-NEXT: return
// INFER-NEXT: }

// Fill in unknown entries without overwriting an already known valid direction.
func.func @partial() {
  %0 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %1 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
  %2 = "daphne.fill"(%1, %0, %0) : (si64, index, index) -> !daphne.Matrix<2x2xsi64:colSortedness[descending, unknown]>
  return
}
// INFER-LABEL: func.func @partial() {
// INFER-NEXT: %0 = "daphne.constant"() <{value = 2 : index}> : () -> index
// INFER-NEXT: %1 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
// INFER-NEXT: %2 = "daphne.fill"(%1, %0, %0) : (si64, index, index) -> !daphne.Matrix<2x2xsi64:symmetric[true]:numDistinct[1]:colSortedness[descending, ascending]>
// INFER-NEXT: return
// INFER-NEXT: }

// A branch merge retains the common first column and discards differing knowledge.
func.func @merge(%arg0: i1, %arg1: !daphne.Matrix<2x2xsi64>) {
  %0 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %1 = "daphne.constant"() <{value = true}> : () -> i1
  %2 = "daphne.constant"() <{value = false}> : () -> i1
  %3 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %4 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
  %5 = scf.if %arg0 -> (!daphne.Matrix<2x2xsi64>) {
    %6 = "daphne.order"(%arg1, %0, %1, %2) : (!daphne.Matrix<2x2xsi64>, index, i1, i1) -> !daphne.Matrix<2x2xsi64>
    scf.yield %6 : !daphne.Matrix<2x2xsi64>
  } else {
    %6 = "daphne.fill"(%4, %3, %3) : (si64, index, index) -> !daphne.Matrix<2x2xsi64>
    scf.yield %6 : !daphne.Matrix<2x2xsi64>
  }
  return
}
// INFER-LABEL: func.func @merge(%arg0: i1, %arg1: !daphne.Matrix<2x2xsi64>) {
// INFER-NEXT: %0 = "daphne.constant"() <{value = 0 : index}> : () -> index
// INFER-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INFER-NEXT: %2 = "daphne.constant"() <{value = false}> : () -> i1
// INFER-NEXT: %3 = "daphne.constant"() <{value = 2 : index}> : () -> index
// INFER-NEXT: %4 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
// INFER-NEXT: %5 = scf.if %arg0 -> (!daphne.Matrix<2x2xsi64:colSortedness[ascending, unknown]>) {
// INFER-NEXT: %6 = "daphne.order"(%arg1, %0, %1, %2) : (!daphne.Matrix<2x2xsi64>, index, i1, i1) -> !daphne.Matrix<2x2xsi64:colSortedness[ascending, unknown]>
// INFER-NEXT: scf.yield %6 : !daphne.Matrix<2x2xsi64:colSortedness[ascending, unknown]>
// INFER-NEXT: } else {
// INFER-NEXT: %6 = "daphne.fill"(%4, %3, %3) : (si64, index, index) -> !daphne.Matrix<2x2xsi64:symmetric[true]:numDistinct[1]:colSortedness[ascending, ascending]>
// INFER-NEXT: %7 = "daphne.cast"(%6) : (!daphne.Matrix<2x2xsi64:symmetric[true]:numDistinct[1]:colSortedness[ascending, ascending]>) -> !daphne.Matrix<2x2xsi64:colSortedness[ascending, unknown]>
// INFER-NEXT: scf.yield %7 : !daphne.Matrix<2x2xsi64:colSortedness[ascending, unknown]>
// INFER-NEXT: }
// INFER-NEXT: return
// INFER-NEXT: }

// A loop that reverses direction cannot retain its initial ordering.
func.func @loop(%arg0: si64) {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index
  %3 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
  %4 = "daphne.fill"(%3, %2, %2) : (si64, index, index) -> !daphne.Matrix<2x2xsi64>
  %5 = scf.for %i = %0 to %2 step %1 iter_args(%matrix = %4) -> (!daphne.Matrix<2x2xsi64>) {
    %6 = "daphne.ewLt"(%matrix, %arg0) : (!daphne.Matrix<2x2xsi64>, si64) -> !daphne.Matrix<2x2xsi64>
    scf.yield %6 : !daphne.Matrix<2x2xsi64>
  }
  return
}
// INFER-LABEL: func.func @loop(%arg0: si64) {
// INFER-NEXT: %c0 = arith.constant 0 : index
// INFER-NEXT: %c1 = arith.constant 1 : index
// INFER-NEXT: %c2 = arith.constant 2 : index
// INFER-NEXT: %0 = "daphne.constant"() <{value = 7 : si64}> : () -> si64
// INFER-NEXT: %1 = "daphne.fill"(%0, %c2, %c2) : (si64, index, index) -> !daphne.Matrix<2x2xsi64:symmetric[true]:numDistinct[1]:colSortedness[ascending, ascending]>
// INFER-NEXT: %2 = "daphne.cast"(%1) : (!daphne.Matrix<2x2xsi64:symmetric[true]:numDistinct[1]:colSortedness[ascending, ascending]>) -> !daphne.Matrix<2x2xsi64:colSortedness[unknown, unknown]>
// INFER-NEXT: %3 = "daphne.cast"(%2) : (!daphne.Matrix<2x2xsi64:colSortedness[unknown, unknown]>) -> !daphne.Matrix<2x2xsi64>
// INFER-NEXT: %4 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %3) -> (!daphne.Matrix<2x2xsi64>) {
// INFER-NEXT: %5 = "daphne.ewLt"(%arg2, %arg0) : (!daphne.Matrix<2x2xsi64>, si64) -> !daphne.Matrix<2x2xsi64:lowerBound[0.000000e+00]:upperBound[1.000000e+00]>
// INFER-NEXT: %6 = "daphne.cast"(%5) : (!daphne.Matrix<2x2xsi64:lowerBound[0.000000e+00]:upperBound[1.000000e+00]>) -> !daphne.Matrix<2x2xsi64>
// INFER-NEXT: scf.yield %6 : !daphne.Matrix<2x2xsi64>
// INFER-NEXT: }
// INFER-NEXT: return
// INFER-NEXT: }

//--- parse.mlir
func.func @columns(%arg0: !daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, %arg1: !daphne.Matrix<?x?xf64:colSortedness[unknown, ascending]>, %arg2: !daphne.Matrix<0x0xf64:colSortedness[]>) { return }
// PARSE-LABEL: func.func @columns(%arg0: !daphne.Matrix<4x3xsi64:colSortedness[ascending, unknown, descending]>, %arg1: !daphne.Matrix<?x?xf64:colSortedness[unknown, ascending]>, %arg2: !daphne.Matrix<0x0xf64>) {
// PARSE-NEXT: return
// PARSE-NEXT: }

//--- invalid.mlir
// expected-error@+1 {{colSortedness must have one entry per column}}
func.func @width(%arg0: !daphne.Matrix<3x2xf64:colSortedness[ascending]>) { return }
// -----
// expected-error@+1 {{invalid column sortedness}}
func.func @direction(%arg0: !daphne.Matrix<3x1xf64:colSortedness[allEqual]>) { return }
// -----
// expected-error@+1 {{duplicate colSortedness}}
func.func @duplicate(%arg0: !daphne.Matrix<3x1xf64:colSortedness[]:colSortedness[ascending]>) { return }

//--- transfer.mlir
func.func @transfer(%arg0: !daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>) {
  %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>) -> !daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>
  return
}
// TRANSFER-LABEL: func.func @transfer(%arg0: !daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>) {
// TRANSFER-NEXT: %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>) -> !daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>
// TRANSFER-NEXT: %1 = "daphne.constant"() <{value = -1.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %2 = "daphne.constant"() <{value = -1 : si64}> : () -> si64
// TRANSFER-NEXT: %3 = "daphne.constant"() <{value = -1 : si64}> : () -> si64
// TRANSFER-NEXT: %4 = "daphne.constant"() <{value = false}> : () -> i1
// TRANSFER-NEXT: %5 = "daphne.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %6 = "daphne.constant"() <{value = false}> : () -> i1
// TRANSFER-NEXT: %7 = "daphne.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %8 = "daphne.constant"() <{value = 1 : si64}> : () -> si64
// TRANSFER-NEXT: %9 = "daphne.constant"() <{value = 2 : si64}> : () -> si64
// TRANSFER-NEXT: "daphne.transferProperties"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9) : (!daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>, f64, si64, si64, i1, f64, i1, f64, si64, si64) -> ()
// TRANSFER-NEXT: return
// TRANSFER-NEXT: }

//--- adapt.mlir
// A value conversion does not establish ordering in the destination value type.
func.func @adapt(%arg0: !daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>, %arg1: f64) {
  %0 = "daphne.ewAdd"(%arg0, %arg1) : (!daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>, f64) -> !daphne.Matrix<3x2xf64>
  return
}
// ADAPT-LABEL: func.func @adapt(%arg0: !daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>, %arg1: f64) {
// ADAPT-NEXT: %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<3x2xsi64:colSortedness[ascending, descending]>) -> !daphne.Matrix<3x2xf64>
// ADAPT-NEXT: %1 = "daphne.ewAdd"(%0, %arg1) : (!daphne.Matrix<3x2xf64>, f64) -> !daphne.Matrix<3x2xf64>
// ADAPT-NEXT: return
// ADAPT-NEXT: }
