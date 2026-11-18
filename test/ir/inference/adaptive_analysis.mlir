// RUN: split-file %s %t
// RUN: daphne-opt %t/requests.mlir --adaptive-analyze-props --cse --verify-each | FileCheck %s --check-prefix=INSERT --implicit-check-not='"daphne.analyzeData"'
// RUN: daphne-opt %t/requests.mlir --pass-pipeline='builtin.module(func.func(adaptive-analyze-props,cse,adaptive-analyze-props,cse))' --verify-each | FileCheck %s --check-prefix=INSERT --implicit-check-not='"daphne.analyzeData"'
// RUN: daphne-opt %t/requests.mlir --verify-each | FileCheck %s --check-prefix=DISABLED

// RUN: daphne-opt %t/local.mlir --adaptive-analyze-props --cse --verify-each | FileCheck %s --check-prefix=LOCAL --implicit-check-not='"daphne.analyzeData"'
// RUN: daphne-opt %t/local.mlir --pass-pipeline='builtin.module(func.func(adaptive-analyze-props,cse,adaptive-analyze-props,cse))' --verify-each | FileCheck %s --check-prefix=LOCAL --implicit-check-not='"daphne.analyzeData"'

//--- requests.mlir
// The interface requests bounds for operand zero, not the scalar.
func.func @comparison(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @comparison(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }


// MatMul reports the lhs independently of transpose flags.
func.func @multiply(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.constant"() <{value = false}> : () -> i1
  %1 = "daphne.matMul"(%arg0, %arg1, %0, %0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>
  return %1 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @multiply(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %1, %0, %0, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.matMul"(%arg0, %arg1, %0, %0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }


func.func @transpose(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>, %arg2: i1) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.matMul"(%arg0, %arg1, %arg2, %arg2) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @transpose(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>, %arg2: i1) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %0, %1, %1, %1) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.matMul"(%arg0, %arg1, %arg2, %arg2) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }


func.func @join(%arg0: !daphne.Column<?xsi64>, %arg1: !daphne.Column<?xsi64>, %arg2: index) -> !daphne.Column<?xui64> {
  %0 = "daphne.colSemiJoin"(%arg0, %arg1, %arg2) : (!daphne.Column<?xsi64>, !daphne.Column<?xsi64>, index) -> !daphne.Column<?xui64>
  return %0 : !daphne.Column<?xui64>
}
// INSERT-LABEL: func.func @join(%arg0: !daphne.Column<?xsi64>, %arg1: !daphne.Column<?xsi64>, %arg2: index) -> !daphne.Column<?xui64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg1, %0, %0, %1, %0) : (!daphne.Column<?xsi64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.colSemiJoin"(%arg0, %arg1, %arg2) : (!daphne.Column<?xsi64>, !daphne.Column<?xsi64>, index) -> !daphne.Column<?xui64>
// INSERT-NEXT: return %2 : !daphne.Column<?xui64>
// INSERT-NEXT: }


// Bounds are requested for both matrix operands, independently of representation.
func.func @matrix_comparison(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @matrix_comparison(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: "daphne.analyzeData"(%arg1, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }


func.func @map_input(%arg0: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.map"(%arg0) <{func = "twice"}> : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @map_input(%arg0: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %0, %0, %0, %1) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.map"(%arg0) <{func = "twice"}> : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }


// No property usage declared for addition.
func.func @ordinary(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.ewAdd"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @ordinary(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.ewAdd"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %0 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }


func.func @branches(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64, %arg2: i1) -> !daphne.Matrix<2x2xf64> {
  %0 = scf.if %arg2 -> (!daphne.Matrix<2x2xf64>) {
    %1 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    scf.yield %1 : !daphne.Matrix<2x2xf64>
  } else {
    %1 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    scf.yield %1 : !daphne.Matrix<2x2xf64>
  }
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @branches(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64, %arg2: i1) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = scf.if %arg2 -> (!daphne.Matrix<2x2xf64>) {
// INSERT-NEXT: %1 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %2 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %1, %1, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %3 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: scf.yield %3 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: } else {
// INSERT-NEXT: %1 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %2 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %1, %1, %2, %1) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %3 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: scf.yield %3 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }
// INSERT-NEXT: return %0 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }

// DISABLED-NOT: "daphne.analyzeData"

// Acquisition also supports explicitly transposed operands.
func.func @known_transpose(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.constant"() <{value = false}> : () -> i1
  %1 = "daphne.constant"() <{value = true}> : () -> i1
  %2 = "daphne.matMul"(%arg0, %arg1, %1, %0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>
  return %2 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @known_transpose(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %1, %0, %0, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.matMul"(%arg0, %arg1, %1, %0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>, i1, i1) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }

// Reversing the comparison requests bounds for operand one, not the scalar.
func.func @reversed_comparison(%arg0: f64, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.ewLt"(%arg0, %arg1) : (f64, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @reversed_comparison(%arg0: f64, %arg1: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg1, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.ewLt"(%arg0, %arg1) : (f64, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }

// Two operand queries for the same SSA value merge into one acquisition.
func.func @shared_operand(%arg0: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
  %0 = "daphne.ewLt"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// INSERT-LABEL: func.func @shared_operand(%arg0: !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64> {
// INSERT-NEXT: %0 = "daphne.constant"() <{value = false}> : () -> i1
// INSERT-NEXT: %1 = "daphne.constant"() <{value = true}> : () -> i1
// INSERT-NEXT: "daphne.analyzeData"(%arg0, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// INSERT-NEXT: %2 = "daphne.ewLt"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// INSERT-NEXT: return %2 : !daphne.Matrix<2x2xf64>
// INSERT-NEXT: }

//--- local.mlir
// The interface declares usage, but these element types have no acquisition support.
func.func @unsupported(%arg0: !daphne.Matrix<2x2xi1>, %arg1: !daphne.Matrix<2x2xindex>) -> (!daphne.Matrix<2x2xi1>, !daphne.Matrix<2x2xindex>) {
  %0 = "daphne.ewEq"(%arg0, %arg0) : (!daphne.Matrix<2x2xi1>, !daphne.Matrix<2x2xi1>) -> !daphne.Matrix<2x2xi1>
  %1 = "daphne.ewEq"(%arg1, %arg1) : (!daphne.Matrix<2x2xindex>, !daphne.Matrix<2x2xindex>) -> !daphne.Matrix<2x2xindex>
  return %0, %1 : !daphne.Matrix<2x2xi1>, !daphne.Matrix<2x2xindex>
}
// LOCAL-LABEL: func.func @unsupported
// LOCAL: return

// A covering acquisition with arith constants is sufficient locally.
// An intervening call ends that local window, even for the same SSA value.
func.func private @write(!daphne.Matrix<2x2xf64:rep[sparse]>)
func.func @local_window(%arg0: !daphne.Matrix<2x2xf64:rep[sparse]>, %arg1: f64, %arg2: f64) -> (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) {
  %true = arith.constant true
  "daphne.analyzeData"(%arg0, %true, %true, %true, %true) : (!daphne.Matrix<2x2xf64:rep[sparse]>, i1, i1, i1, i1) -> ()
  %false = arith.constant false
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64:rep[sparse]>, f64) -> !daphne.Matrix<2x2xf64>
  func.call @write(%arg0) : (!daphne.Matrix<2x2xf64:rep[sparse]>) -> ()
  %1 = "daphne.ewLt"(%arg0, %arg2) : (!daphne.Matrix<2x2xf64:rep[sparse]>, f64) -> !daphne.Matrix<2x2xf64>
  return %0, %1 : !daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>
}
// LOCAL-LABEL: func.func @local_window
// LOCAL: "daphne.analyzeData"(%arg0, %true, %true, %true, %true) : (!daphne.Matrix<2x2xf64:rep[sparse]>, i1, i1, i1, i1) -> ()
// LOCAL: "daphne.ewLt"
// LOCAL: call @write
// LOCAL: %1 = "daphne.constant"() <{value = false}> : () -> i1
// LOCAL: %2 = "daphne.constant"() <{value = true}> : () -> i1
// LOCAL: "daphne.analyzeData"(%arg0, %1, %1, %2, %1) : (!daphne.Matrix<2x2xf64:rep[sparse]>, i1, i1, i1, i1) -> ()
// LOCAL: "daphne.ewLt"
// LOCAL: return

// Sequential loops keep acquisition in the loop; parallel regions are not traversed.
func.func private @consume(!daphne.Matrix<2x2xf64>)
func.func @loops(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %arg2 step %c1 {
    %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    func.call @consume(%0) : (!daphne.Matrix<2x2xf64>) -> ()
  }
  scf.parallel (%i) = (%c0) to (%arg2) step (%c1) {
    %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
    func.call @consume(%0) : (!daphne.Matrix<2x2xf64>) -> ()
    scf.reduce
  }
  return
}
// LOCAL-LABEL: func.func @loops
// LOCAL: scf.for
// LOCAL: %0 = "daphne.constant"() <{value = false}> : () -> i1
// LOCAL: %1 = "daphne.constant"() <{value = true}> : () -> i1
// LOCAL: "daphne.analyzeData"(%arg0, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// LOCAL: "daphne.ewLt"
// LOCAL: scf.parallel
// LOCAL: "daphne.ewLt"
// LOCAL: return

// An unrelated explicit request must not suppress the requested bounds scan.
func.func @partial(%arg0: !daphne.Matrix<2x2xf64>, %arg1: f64) -> !daphne.Matrix<2x2xf64> {
  %true = arith.constant true
  %false = arith.constant false
  "daphne.analyzeData"(%arg0, %true, %false, %false, %false) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
  return %0 : !daphne.Matrix<2x2xf64>
}
// LOCAL-LABEL: func.func @partial
// LOCAL: "daphne.analyzeData"(%arg0, %true, %false, %false, %false) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// LOCAL: %0 = "daphne.constant"() <{value = false}> : () -> i1
// LOCAL: %1 = "daphne.constant"() <{value = true}> : () -> i1
// LOCAL: "daphne.analyzeData"(%arg0, %0, %0, %1, %0) : (!daphne.Matrix<2x2xf64>, i1, i1, i1, i1) -> ()
// LOCAL: %2 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64>, f64) -> !daphne.Matrix<2x2xf64>
// LOCAL: return %2 : !daphne.Matrix<2x2xf64>
