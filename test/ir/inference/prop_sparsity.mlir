// RUN: split-file %s %t
// RUN: daphne-opt %t/rules.mlir --inference --verify-each | daphne-opt --verify-each | FileCheck %s
// RUN: daphne-opt %t/diagonal.mlir --inference --verify-each | FileCheck %s --check-prefix=ENABLED
// RUN: daphne-opt %t/types.mlir --inference --verify-each | FileCheck %s --check-prefix=TYPES
// RUN: daphne-opt %t/casts.mlir --record-properties --verify-each | FileCheck %s --check-prefix=CASTS
// RUN: daphne-opt %t/invalid.mlir --split-input-file --verify-diagnostics

//--- rules.mlir
// A classification is not a layout identity. Only floating diagonal classifications propagate.
func.func @f64_unknown(%arg0: !daphne.Matrix<2x2xf64>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  return
}
// CHECK-LABEL: func.func @f64_unknown(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>

func.func @f64_0(%arg0: !daphne.Matrix<2x2xf64:pattern[0]>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  return
}
// CHECK-LABEL: func.func @f64_0(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64:pattern[0]>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64:pattern[0]>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64:pattern[0]>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64:pattern[0]>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64:pattern[0]>

func.func @f64_1(%arg0: !daphne.Matrix<2x2xf64:pattern[1]>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>, !daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>, !daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>, !daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
  return
}
// CHECK-LABEL: func.func @f64_1(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>, !daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>, !daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[1]>, !daphne.Matrix<2x2xf64:pattern[1]>) -> !daphne.Matrix<2x2xf64>

func.func @f64_11(%arg0: !daphne.Matrix<2x2xf64:pattern[11]>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>, !daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>, !daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>, !daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
  return
}
// CHECK-LABEL: func.func @f64_11(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>, !daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>, !daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xf64:pattern[11]>, !daphne.Matrix<2x2xf64:pattern[11]>) -> !daphne.Matrix<2x2xf64>

func.func @si64_unknown(%arg0: !daphne.Matrix<2x2xsi64>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64>, !daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64>, !daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64>, !daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
  return
}
// CHECK-LABEL: func.func @si64_unknown(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64>, !daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64>, !daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64>, !daphne.Matrix<2x2xsi64>) -> !daphne.Matrix<2x2xsi64>

func.func @si64_0(%arg0: !daphne.Matrix<2x2xsi64:pattern[0]>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>, !daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>, !daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>, !daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
  return
}
// CHECK-LABEL: func.func @si64_0(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>, !daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>, !daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[0]>, !daphne.Matrix<2x2xsi64:pattern[0]>) -> !daphne.Matrix<2x2xsi64>

func.func @si64_1(%arg0: !daphne.Matrix<2x2xsi64:pattern[1]>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>, !daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>, !daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>, !daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
  return
}
// CHECK-LABEL: func.func @si64_1(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>, !daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>, !daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[1]>, !daphne.Matrix<2x2xsi64:pattern[1]>) -> !daphne.Matrix<2x2xsi64>

func.func @si64_11(%arg0: !daphne.Matrix<2x2xsi64:pattern[11]>) {
  %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
  %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
  %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
  %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>, !daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
  %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>, !daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
  %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>, !daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
  return
}
// CHECK-LABEL: func.func @si64_11(
// CHECK: %0 = "daphne.ewAbs"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %1 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %2 = "daphne.ewLn"(%arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %3 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>, !daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %4 = "daphne.ewSub"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>, !daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>
// CHECK: %5 = "daphne.ewMul"(%arg0, %arg0) : (!daphne.Matrix<2x2xsi64:pattern[11]>, !daphne.Matrix<2x2xsi64:pattern[11]>) -> !daphne.Matrix<2x2xsi64>

func.func @operands(%arg0: !daphne.Matrix<2x2xf64:pattern[0]>, %arg1: !daphne.Matrix<2x2xf64>, %arg2: !daphne.Matrix<1x2xf64:pattern[0]>, %arg3: !daphne.Matrix<?x2xf64:pattern[0]>) {
  %0 = "daphne.ewMul"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
  %1 = "daphne.ewMul"(%arg0, %arg2) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<1x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %2 = "daphne.ewMul"(%arg0, %arg3) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<?x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  return
}
// CHECK-LABEL: func.func @operands(
// CHECK: %0 = "daphne.ewMul"(%arg0, %arg1) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// CHECK: %1 = "daphne.ewMul"(%arg0, %arg2) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<1x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
// CHECK: %2 = "daphne.ewMul"(%arg0, %arg3) : (!daphne.Matrix<2x2xf64:pattern[0]>, !daphne.Matrix<?x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>

//--- diagonal.mlir
func.func @diagonal(%arg0: !daphne.Matrix<3x1xf64>) {
  %0 = "daphne.diagMatrix"(%arg0) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
  %1 = "daphne.ewAbs"(%0) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x3xf64>
  return
}
// ENABLED: %0 = "daphne.diagMatrix"(%arg0) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64:sp[0.33333333333333331]:pattern[0]>
// ENABLED: %1 = "daphne.ewAbs"(%0) : (!daphne.Matrix<3x3xf64:sp[0.33333333333333331]:pattern[0]>) -> !daphne.Matrix<3x3xf64:pattern[0]>

//--- types.mlir
// Inference rebuilds result types: changed shape/element type clears classifications.
func.func @types(%arg0: !daphne.Matrix<2x2xf64>, %arg1: !daphne.Matrix<?x2xf64>) {
  %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<?x2xf64:pattern[0]>
  %1 = "daphne.cast"(%arg1) : (!daphne.Matrix<?x2xf64>) -> !daphne.Matrix<?x2xf64:pattern[0]>
  %2 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2x!daphne.Unknown:pattern[0]>
  %3 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64:pattern[0]>
  return
}
// TYPES-LABEL: func.func @types(
// TYPES: %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// TYPES: %1 = "daphne.cast"(%arg1) : (!daphne.Matrix<?x2xf64>) -> !daphne.Matrix<?x2xf64:pattern[0]>
// TYPES: %2 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64>
// TYPES: %3 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64:pattern[0]>

//--- casts.mlir
// Property removal is skipped by recording; adding a classification is not.
func.func @main(%arg0: !daphne.Matrix<2x2xf64:pattern[0]>, %arg1: !daphne.Matrix<2x2xf64>) {
  %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
  %1 = "daphne.cast"(%arg1) : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64:pattern[0]>
  return
}
// CASTS-LABEL: func.func @main(
// CASTS: %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x2xf64:pattern[0]>) -> !daphne.Matrix<2x2xf64>
// CASTS-NEXT: %1 = "daphne.cast"(%arg1) {daphne.value_ids = [1 : ui32]} : (!daphne.Matrix<2x2xf64>) -> !daphne.Matrix<2x2xf64:pattern[0]>
// CASTS-NEXT: %2 = "daphne.constant"() <{value = 1 : ui32}> : () -> ui32
// CASTS-NEXT: "daphne.recordProperties"(%1, %2) : (!daphne.Matrix<2x2xf64:pattern[0]>, ui32) -> ()
// CASTS-NEXT: return

//--- invalid.mlir
// expected-error@+1 {{sparsity pattern classification must be nonnegative}}
func.func @negative(%arg0: !daphne.Matrix<2x2xf64:pattern[-2]>) { return }
// -----
// expected-error@+1 {{duplicate sparsity pattern classification}}
func.func @duplicate(%arg0: !daphne.Matrix<2x2xf64:pattern[0]:pattern[1]>) { return }
