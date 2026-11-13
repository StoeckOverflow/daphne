// RUN: daphne-opt --inference %s --verify-each | FileCheck %s

// Original sequence inference: integer directions, floating steps, unknown operands.
func.func @seq_rules(%unknown: f64) {
  %c0 = "daphne.constant"() <{value = 0 : si64}> : () -> si64
  %c1 = "daphne.constant"() <{value = 10 : si64}> : () -> si64
  %c2 = "daphne.constant"() <{value = 2 : si64}> : () -> si64
  %ascending = "daphne.seq"(%c0, %c1, %c2) : (si64, si64, si64) -> !daphne.Matrix<?x1xsi64>
  %c3 = "daphne.constant"() <{value = -2 : si64}> : () -> si64
  %descending = "daphne.seq"(%c1, %c0, %c3) : (si64, si64, si64) -> !daphne.Matrix<?x1xsi64>
  %f0 = "daphne.constant"() <{value = 0.0 : f64}> : () -> f64
  %f1 = "daphne.constant"() <{value = 1.0 : f64}> : () -> f64
  %f2 = "daphne.constant"() <{value = 0.25 : f64}> : () -> f64
  %fractional = "daphne.seq"(%f0, %f1, %f2) : (f64, f64, f64) -> !daphne.Matrix<?x1xf64>
  %unknown_sequence = "daphne.seq"(%f0, %unknown, %f2) : (f64, f64, f64) -> !daphne.Matrix<?x1xf64>
  %s0 = "daphne.constant"() <{value = 0.0 : f32}> : () -> f32
  %s1 = "daphne.constant"() <{value = 1.0 : f32}> : () -> f32
  %s2 = "daphne.constant"() <{value = 0.25 : f32}> : () -> f32
  %single_precision = "daphne.seq"(%s0, %s1, %s2) : (f32, f32, f32) -> !daphne.Matrix<?x1xf32>
  return
}
// CHECK-LABEL: func.func @seq_rules(%arg0: f64) {
// CHECK: %3 = "daphne.seq"(%0, %1, %2) : (si64, si64, si64) -> !daphne.Matrix<6x1xsi64:sp[1.000000e+00]:dist[6]>
// CHECK: %5 = "daphne.seq"(%1, %0, %4) : (si64, si64, si64) -> !daphne.Matrix<6x1xsi64:sp[1.000000e+00]:dist[6]>
// CHECK: %9 = "daphne.seq"(%6, %7, %8) : (f64, f64, f64) -> !daphne.Matrix<5x1xf64:sp[1.000000e+00]:dist[5]>
// CHECK: %10 = "daphne.seq"(%6, %arg0, %8) : (f64, f64, f64) -> !daphne.Matrix<?x1xf64:sp[1.000000e+00]>
// CHECK: %14 = "daphne.seq"(%11, %12, %13) : (f32, f32, f32) -> !daphne.Matrix<5x1xf32:sp[1.000000e+00]:dist[5]>

func.func @fill_rules(%n: index, %unknown: f64) {
  // negative zero
  %c2 = "daphne.constant"() <{value = -0.0 : f64}> : () -> f64
  %c1 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %fill_negative_zero = "daphne.fill"(%c2, %c1, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // nan
  %c4 = "daphne.constant"() <{value = 0x7FF8000000000000 : f64}> : () -> f64
  %fill_nan = "daphne.fill"(%c4, %c1, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // empty rows
  %c13 = "daphne.constant"() <{value = 2.0 : f64}> : () -> f64
  %c14 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %fill_empty_rows = "daphne.fill"(%c13, %c14, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // unknown rows
  %fill_unknown_rows = "daphne.fill"(%c13, %n, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // nonconstant
  %fill_nonconstant = "daphne.fill"(%unknown, %c1, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  return
}

// CHECK-LABEL: func.func @fill_rules(%arg0: index, %arg1: f64) {
// CHECK: %2 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]:dist[1]>
// CHECK: %4 = "daphne.fill"(%3, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
// CHECK: %7 = "daphne.fill"(%5, %6, %1) : (f64, index, index) -> !daphne.Matrix<0x2xf64:symmetric[false]:dist[0]>
// CHECK: %8 = "daphne.fill"(%5, %arg0, %1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
// CHECK: %9 = "daphne.fill"(%arg1, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
