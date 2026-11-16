/*
 * Copyright 2025 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ir/daphneir/Daphne.h>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSparsityPatternOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;

// ****************************************************************************
// Compile-time pattern ID constants
// ****************************************************************************

// A compile-time classification is not a runtime CSR-layout identity.
// Diagonal means every off-diagonal value is zero. Diagonal entries may also
// be zero or nonfinite.
namespace {
constexpr ssize_t PATTERN_ID_UNKNOWN = -1;
constexpr ssize_t PATTERN_ID_DIAGONAL = 0;

bool isFloatingDiagonal(daphne::MatrixType type) {
    return type && type.getSparsityPatternID() == PATTERN_ID_DIAGONAL &&
           (type.getElementType().isF32() || type.getElementType().isF64());
}
} // namespace

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<ssize_t> daphne::DiagMatrixOp::inferSparsityPattern() { return {PATTERN_ID_DIAGONAL}; }

// ****************************************************************************
// Unary operations: Pattern propagation
// ****************************************************************************

// Only propagate the supported classification for floating-point operations.
std::vector<ssize_t> daphne::EwSqrtOp::inferSparsityPattern() {
    auto inputMatrix = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    if (!inputMatrix)
        return {PATTERN_ID_UNKNOWN};

    return {isFloatingDiagonal(inputMatrix) ? PATTERN_ID_DIAGONAL : PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwLnOp::inferSparsityPattern() {
    auto inputMatrix = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    if (!inputMatrix)
        return {PATTERN_ID_UNKNOWN};

    return {PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwAbsOp::inferSparsityPattern() {
    auto inputMatrix = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    if (!inputMatrix)
        return {PATTERN_ID_UNKNOWN};

    return {isFloatingDiagonal(inputMatrix) ? PATTERN_ID_DIAGONAL : PATTERN_ID_UNKNOWN};
}

// ****************************************************************************
// Binary operations: Pattern combination
// ****************************************************************************

std::vector<ssize_t> daphne::EwAddOp::inferSparsityPattern() {
    auto lhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    if (!lhsMatrix || !rhsMatrix)
        return {PATTERN_ID_UNKNOWN};

    // Both off-diagonal operands are zero, including for multiplication.
    // Do not infer from one diagonal operand: 0 * infinity can produce NaN.
    // Require equal known shapes rather than assuming broadcasting preserves it.
    if (isFloatingDiagonal(lhsMatrix) && isFloatingDiagonal(rhsMatrix) &&
        lhsMatrix.getElementType() == rhsMatrix.getElementType() && lhsMatrix.getNumRows() >= 0 &&
        lhsMatrix.getNumCols() >= 0 && lhsMatrix.getNumRows() == rhsMatrix.getNumRows() &&
        lhsMatrix.getNumCols() == rhsMatrix.getNumCols())
        return {PATTERN_ID_DIAGONAL};

    return {PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwSubOp::inferSparsityPattern() {
    auto lhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    if (!lhsMatrix || !rhsMatrix)
        return {PATTERN_ID_UNKNOWN};

    if (isFloatingDiagonal(lhsMatrix) && isFloatingDiagonal(rhsMatrix) &&
        lhsMatrix.getElementType() == rhsMatrix.getElementType() && lhsMatrix.getNumRows() >= 0 &&
        lhsMatrix.getNumCols() >= 0 && lhsMatrix.getNumRows() == rhsMatrix.getNumRows() &&
        lhsMatrix.getNumCols() == rhsMatrix.getNumCols())
        return {PATTERN_ID_DIAGONAL};

    return {PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwMulOp::inferSparsityPattern() {
    auto lhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    if (!lhsMatrix || !rhsMatrix)
        return {PATTERN_ID_UNKNOWN};

    if (isFloatingDiagonal(lhsMatrix) && isFloatingDiagonal(rhsMatrix) &&
        lhsMatrix.getElementType() == rhsMatrix.getElementType() && lhsMatrix.getNumRows() >= 0 &&
        lhsMatrix.getNumCols() >= 0 && lhsMatrix.getNumRows() == rhsMatrix.getNumRows() &&
        lhsMatrix.getNumCols() == rhsMatrix.getNumCols())
        return {PATTERN_ID_DIAGONAL};

    return {PATTERN_ID_UNKNOWN};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<ssize_t> daphne::tryInferSparsityPattern(Operation *op) {
    if (auto inferPatternOp = llvm::dyn_cast<InferSparsityPattern>(op))
        // If the operation implements the inference interface, we apply that.
        return inferPatternOp.inferSparsityPattern();
    else {
        // If the operation does not implement the inference interface
        // we return unknown.
        std::vector<ssize_t> patternIDs;
        for (size_t i = 0; i < op->getNumResults(); i++)
            patternIDs.push_back(PATTERN_ID_UNKNOWN);
        return patternIDs;
    }
}
