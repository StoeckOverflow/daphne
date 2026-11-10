// RUN: daphne-opt %s --split-input-file --verify-diagnostics

func.func @invalid_bits(%arg: !daphne.Matrix<2x2xf64>) {
  // expected-error @+1 {{property 'requested_properties' failed to satisfy constraint: requested data properties}}
  "daphne.analyzeData"(%arg) <{requested_properties = 64 : i32}> : (!daphne.Matrix<2x2xf64>) -> ()
  return
}

// -----

func.func @invalid_input(%arg: f64) {
  // expected-error @+1 {{operand #0 must be}}
  "daphne.analyzeData"(%arg) <{requested_properties = 1 : i32}> : (f64) -> ()
  return
}
