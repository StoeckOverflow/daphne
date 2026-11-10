// RUN: daphne-opt %s | FileCheck %s
// RUN: daphne-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

func.func @requests(%arg: !daphne.Matrix<2x2xf64>) {
  daphne.analyzeData %arg requested_properties = none : !daphne.Matrix<2x2xf64>
  daphne.analyzeData %arg requested_properties = sparsity : !daphne.Matrix<2x2xf64>
  daphne.analyzeData %arg requested_properties = "minmax|distinct" : !daphne.Matrix<2x2xf64>
  daphne.analyzeData %arg requested_properties = all : !daphne.Matrix<2x2xf64>
  return
}
// CHECK: requested_properties = none
// CHECK: requested_properties = sparsity
// CHECK: requested_properties = "minmax|distinct"
// CHECK: requested_properties = all
// GENERIC: "daphne.analyzeData"(%arg0) <{requested_properties = 0 : i32}>
// GENERIC: "daphne.analyzeData"(%arg0) <{requested_properties = 1 : i32}>
// GENERIC: "daphne.analyzeData"(%arg0) <{requested_properties = 24 : i32}>
// GENERIC: "daphne.analyzeData"(%arg0) <{requested_properties = 63 : i32}>
