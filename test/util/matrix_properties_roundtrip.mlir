// RUN: daphne-opt %s > %t
// RUN: FileCheck %s < %t
// RUN: daphne-opt %t > %t.roundtrip
// RUN: diff %t %t.roundtrip

func.func private @unknown(!daphne.Matrix<2x2xf64>)
func.func private @known(!daphne.Matrix<2x2xf64:symmetric[true]:sortness[sortedAsc]:min[-1.0]:max[4.0]:dist[3]>)
func.func private @lower_bound(!daphne.Matrix<?x?xf64:min[-1.0]>)
func.func private @upper_bound(!daphne.Matrix<?x?xf64:max[4.0]>)
func.func private @empty(!daphne.Matrix<0x0xf64:dist[0]>)

// CHECK: @unknown(!daphne.Matrix<2x2xf64>)
// CHECK: @known(!daphne.Matrix<2x2xf64:symmetric[true]:sortness[sortedAsc]:min[{{[^]]+}}]:max[{{[^]]+}}]:dist[3]>)
// CHECK: @lower_bound(!daphne.Matrix<?x?xf64:min[{{[^]]+}}]>)
// CHECK: @upper_bound(!daphne.Matrix<?x?xf64:max[{{[^]]+}}]>)
// CHECK: @empty(!daphne.Matrix<0x0xf64:dist[0]>)
