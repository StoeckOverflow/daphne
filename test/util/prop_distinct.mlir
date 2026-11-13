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
