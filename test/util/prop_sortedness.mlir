// RUN: daphne-opt %s --verify-each | daphne-opt --verify-each | FileCheck %s --check-prefix=BASELINE
// RUN: daphne-opt %s --inference --verify-each | FileCheck %s --check-prefix=INFER

// Preserve an explicit sortedness annotation when printing and parsing.
func.func @sorted(%arg0: !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) {
  %0 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64>
  return
}
// BASELINE-LABEL: func.func @sorted(%arg0: !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) {
// BASELINE-NEXT: %0 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64>
// INFER-LABEL: func.func @sorted(%arg0: !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) {
// INFER-NEXT: %0 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>

// Under order-preserving arithmetic, matching directions and constant offsets preserve order
func.func @addition(%arg0: !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, %arg1: !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, %arg2: !daphne.Matrix<3x1xf64>, %arg3: !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) {
  %0 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>) -> !daphne.Matrix<3x1xf64>
  %1 = "daphne.ewAdd"(%arg0, %arg1) : (!daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64>
  %2 = "daphne.ewAdd"(%arg1, %arg2) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64>
  %3 = "daphne.ewAdd"(%arg3, %arg3) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) -> !daphne.Matrix<3x1xf64>
  %4 = "daphne.ewAdd"(%arg3, %arg1) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64>
  %5 = "daphne.ewAdd"(%arg1, %arg3) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) -> !daphne.Matrix<3x1xf64>
  %6 = "daphne.ewAdd"(%arg3, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>) -> !daphne.Matrix<3x1xf64>
  %7 = "daphne.ewAdd"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) -> !daphne.Matrix<3x1xf64>
  %8 = "daphne.ewAdd"(%arg3, %arg2) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64>
  return
}
// BASELINE-LABEL: func.func @addition(%arg0: !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, %arg1: !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, %arg2: !daphne.Matrix<3x1xf64>, %arg3: !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) {
// INFER-LABEL: func.func @addition(
// INFER-NEXT: %0 = "daphne.ewAdd"(%arg0, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>) -> !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>
// INFER-NEXT: %1 = "daphne.ewAdd"(%arg0, %arg1) : (!daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64>
// INFER-NEXT: %2 = "daphne.ewAdd"(%arg1, %arg2) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64>

// INFER-NEXT: %3 = "daphne.ewAdd"(%arg3, %arg3) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) -> !daphne.Matrix<3x1xf64:sortedness[AllEqual]>
// INFER-NEXT: %4 = "daphne.ewAdd"(%arg3, %arg1) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>) -> !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>
// INFER-NEXT: %5 = "daphne.ewAdd"(%arg1, %arg3) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) -> !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>
// INFER-NEXT: %6 = "daphne.ewAdd"(%arg3, %arg0) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>) -> !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>
// INFER-NEXT: %7 = "daphne.ewAdd"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xf64:sortedness[AllEqual]>) -> !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>
// INFER-NEXT: %8 = "daphne.ewAdd"(%arg3, %arg2) : (!daphne.Matrix<3x1xf64:sortedness[AllEqual]>, !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64>

// Sort values, not permutation indices. A dynamic direction stays unknown.
func.func @ordering(%arg0: !daphne.Matrix<3x1xf64>, %arg1: i1) {
  %0 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %1 = "daphne.constant"() <{value = true}> : () -> i1
  %2 = "daphne.constant"() <{value = false}> : () -> i1
  %3 = "daphne.order"(%arg0, %0, %1, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64>
  %4 = "daphne.order"(%arg0, %0, %2, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64>
  %5 = "daphne.order"(%arg0, %0, %arg1, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64>
  return
}
// INFER-LABEL: func.func @ordering(
// INFER: %3 = "daphne.order"(%arg0, %0, %1, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>
// INFER-NEXT: %4 = "daphne.order"(%arg0, %0, %2, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64:sortedness[sortedDesc]>
// INFER-NEXT: %5 = "daphne.order"(%arg0, %0, %arg1, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64>

// Thresholds outside the bounds yield AllEqual even for unordered inputs
func.func @comparison(%arg0: !daphne.Matrix<3x1xsi64:min[2.0]:max[5.0]>, %arg1: !daphne.Matrix<3x1xsi64>, %arg2: si64) {
  %0 = "daphne.constant"() <{value = 1 : si64}> : () -> si64
  %1 = "daphne.constant"() <{value = 6 : si64}> : () -> si64
  %2 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xsi64:min[2.0]:max[5.0]>, si64) -> !daphne.Matrix<3x1xsi64>
  %3 = "daphne.ewLt"(%arg0, %1) : (!daphne.Matrix<3x1xsi64:min[2.0]:max[5.0]>, si64) -> !daphne.Matrix<3x1xsi64>
  %4 = "daphne.ewLt"(%arg1, %1) : (!daphne.Matrix<3x1xsi64>, si64) -> !daphne.Matrix<3x1xsi64>
  %5 = "daphne.ewLt"(%arg0, %arg2) : (!daphne.Matrix<3x1xsi64:min[2.0]:max[5.0]>, si64) -> !daphne.Matrix<3x1xsi64>
  return
}
// INFER-LABEL: func.func @comparison(
// INFER: %2 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xsi64:min[2.000000e+00]:max[5.000000e+00]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[AllEqual]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %3 = "daphne.ewLt"(%arg0, %1) : (!daphne.Matrix<3x1xsi64:min[2.000000e+00]:max[5.000000e+00]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[AllEqual]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %4 = "daphne.ewLt"(%arg1, %1) : (!daphne.Matrix<3x1xsi64>, si64) -> !daphne.Matrix<3x1xsi64:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %5 = "daphne.ewLt"(%arg0, %arg2) : (!daphne.Matrix<3x1xsi64:min[2.000000e+00]:max[5.000000e+00]>, si64) -> !daphne.Matrix<3x1xsi64:min[0.000000e+00]:max[1.000000e+00]>

// Index output, unknown result mode, multiple columns and missing keys establish no ordering
func.func @unsupported_order(%arg0: !daphne.Matrix<3x1xsi64>, %arg1: !daphne.Matrix<3x2xsi64>, %arg2: i1, %arg3: index) {
  %0 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %1 = "daphne.constant"() <{value = true}> : () -> i1
  %2 = "daphne.constant"() <{value = false}> : () -> i1
  %3 = "daphne.order"(%arg0, %0, %1, %1) : (!daphne.Matrix<3x1xsi64>, index, i1, i1) -> !daphne.Matrix<?x?xindex>
  %4 = "daphne.order"(%arg0, %0, %1, %arg2) : (!daphne.Matrix<3x1xsi64>, index, i1, i1) -> !daphne.Matrix<?x?xsi64>
  %5 = "daphne.order"(%arg1, %0, %1, %2) : (!daphne.Matrix<3x2xsi64>, index, i1, i1) -> !daphne.Matrix<?x?xsi64>
  %6 = "daphne.order"(%arg0, %2) : (!daphne.Matrix<3x1xsi64>, i1) -> !daphne.Matrix<?x?xsi64>
  %7 = "daphne.order"(%arg0, %arg3, %1, %2) : (!daphne.Matrix<3x1xsi64>, index, i1, i1) -> !daphne.Matrix<?x?xsi64>
  return
}
// INFER-LABEL: func.func @unsupported_order(
// INFER: %3 = "daphne.order"(%arg0, %0, %1, %1) : (!daphne.Matrix<3x1xsi64>, index, i1, i1) -> !daphne.Matrix<3x1xindex>
// INFER-NEXT: %4 = "daphne.order"(%arg0, %0, %1, %arg2) : (!daphne.Matrix<3x1xsi64>, index, i1, i1) -> !daphne.Matrix<3x?xsi64>
// INFER-NEXT: %5 = "daphne.order"(%arg1, %0, %1, %2) : (!daphne.Matrix<3x2xsi64>, index, i1, i1) -> !daphne.Matrix<3x2xsi64>
// INFER-NEXT: %6 = "daphne.order"(%arg0, %2) : (!daphne.Matrix<3x1xsi64>, i1) -> !daphne.Matrix<3x1xsi64>
// INFER-NEXT: %7 = "daphne.order"(%arg0, %arg3, %1, %2) : (!daphne.Matrix<3x1xsi64>, index, i1, i1) -> !daphne.Matrix<3x1xsi64>

// Interior thresholds reverse direction, Dynamic integer thresholds do too
func.func @directions(%arg0: !daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[1.0]:max[3.0]>, %arg1: !daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, %arg2: !daphne.Matrix<3x1xsi64:sortedness[AllEqual]>, %arg3: si64) {
  %0 = "daphne.constant"() <{value = 2 : si64}> : () -> si64
  %1 = "daphne.constant"() <{value = 1 : si64}> : () -> si64
  %2 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[1.0]:max[3.0]>, si64) -> !daphne.Matrix<3x1xsi64>
  %3 = "daphne.ewLt"(%arg1, %0) : (!daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, si64) -> !daphne.Matrix<3x1xsi64>
  %4 = "daphne.ewLt"(%arg0, %1) : (!daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[1.0]:max[3.0]>, si64) -> !daphne.Matrix<3x1xsi64>
  %5 = "daphne.ewLt"(%arg1, %arg3) : (!daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, si64) -> !daphne.Matrix<3x1xsi64>
  %6 = "daphne.ewLt"(%arg2, %arg3) : (!daphne.Matrix<3x1xsi64:sortedness[AllEqual]>, si64) -> !daphne.Matrix<3x1xsi64>
  %7 = "daphne.ewAdd"(%arg1, %arg1) : (!daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>) -> !daphne.Matrix<3x1xsi64>
  return
}
// INFER-LABEL: func.func @directions(
// INFER: %2 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[1.000000e+00]:max[3.000000e+00]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[sortedDesc]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %3 = "daphne.ewLt"(%arg1, %0) : (!daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %4 = "daphne.ewLt"(%arg0, %1) : (!daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[1.000000e+00]:max[3.000000e+00]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[AllEqual]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %5 = "daphne.ewLt"(%arg1, %arg3) : (!daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[sortedAsc]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %6 = "daphne.ewLt"(%arg2, %arg3) : (!daphne.Matrix<3x1xsi64:sortedness[AllEqual]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[AllEqual]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %7 = "daphne.ewAdd"(%arg1, %arg1) : (!daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>, !daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>) -> !daphne.Matrix<3x1xsi64:sortedness[sortedDesc]>

// Fractional and large bounds are compared without truncating them to integers.
// Floating-point comparison ordering also stays conservative.
func.func @numeric_bounds(%arg0: !daphne.Matrix<3x1xsi64:min[1.5]:max[4.5]>, %arg1: !daphne.Matrix<3x1xsi64:min[-1.0e100]:max[1.0e100]>, %arg2: !daphne.Matrix<3x1xsi64:min[9.007199254740992e15]:max[9.007199254740992e15]>, %arg3: !daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, %arg4: f64) {
  %0 = "daphne.constant"() <{value = 1 : si64}> : () -> si64
  %1 = "daphne.constant"() <{value = 9007199254740993 : si64}> : () -> si64
  %2 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xsi64:min[1.5]:max[4.5]>, si64) -> !daphne.Matrix<3x1xsi64>
  %3 = "daphne.ewLt"(%arg1, %0) : (!daphne.Matrix<3x1xsi64:min[-1.0e100]:max[1.0e100]>, si64) -> !daphne.Matrix<3x1xsi64>
  %4 = "daphne.ewLt"(%arg2, %1) : (!daphne.Matrix<3x1xsi64:min[9.007199254740992e15]:max[9.007199254740992e15]>, si64) -> !daphne.Matrix<3x1xsi64>
  %5 = "daphne.ewLt"(%arg3, %arg4) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, f64) -> !daphne.Matrix<3x1xf64>
  return
}
// INFER-LABEL: func.func @numeric_bounds(
// INFER: %2 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xsi64:min[1.500000e+00]:max[4.500000e+00]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[AllEqual]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %3 = "daphne.ewLt"(%arg1, %0) : (!daphne.Matrix<3x1xsi64:min[-1.000000e+100]:max[1.000000e+100]>, si64) -> !daphne.Matrix<3x1xsi64:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %4 = "daphne.ewLt"(%arg2, %1) : (!daphne.Matrix<3x1xsi64:min[0x4340000000000000]:max[0x4340000000000000]>, si64) -> !daphne.Matrix<3x1xsi64:sortedness[AllEqual]:min[0.000000e+00]:max[1.000000e+00]>
// INFER-NEXT: %5 = "daphne.ewLt"(%arg3, %arg4) : (!daphne.Matrix<3x1xf64:sortedness[sortedAsc]>, f64) -> !daphne.Matrix<3x1xf64:min[0.000000e+00]:max[1.000000e+00]>
