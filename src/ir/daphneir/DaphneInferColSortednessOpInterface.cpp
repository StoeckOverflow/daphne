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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferColSortednessOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::daphne;

std::vector<std::vector<ColSortedness>> OrderOp::inferColSortedness() {
    auto matrix = dyn_cast<MatrixType>(getArg().getType());
    auto [knownResult, returnIndices] = CompilerUtils::isConstant<bool>(getReturnIdxs());
    if (!matrix || matrix.getNumCols() < 0 || !isa<IntegerType>(matrix.getElementType()) || !knownResult ||
        returnIndices || getColIdxs().empty() || getAscs().empty())
        return {{}};
    auto [knownKey, colIdx] = CompilerUtils::isConstant<size_t>(getColIdxs()[0]);
    auto [knownDirection, ascending] = CompilerUtils::isConstant<bool>(getAscs()[0]);
    if (!knownKey || colIdx >= static_cast<size_t>(matrix.getNumCols()) || !knownDirection)
        return {{}};
    // Only the first lexicographic key is globally ordered down its column.
    std::vector<ColSortedness> colSortedness(matrix.getNumCols(), ColSortedness::Unknown);
    colSortedness[colIdx] = ascending ? ColSortedness::Ascending : ColSortedness::Descending;
    return {colSortedness};
}

std::vector<std::vector<ColSortedness>> EwLtOp::inferColSortedness() {
    auto matrix = dyn_cast<MatrixType>(getLhs().getType());
    // Matching signed integers avoid NaNs and mixed-type conversions.
    if (!matrix || !matrix.getElementType().isSignedInteger(64) || getRhs().getType() != matrix.getElementType())
        return {{}};
    std::vector<ColSortedness> colSortedness(matrix.getColSortedness().begin(), matrix.getColSortedness().end());
    for (auto &direction : colSortedness) {
        if (direction == ColSortedness::Ascending)
            direction = ColSortedness::Descending;
        else if (direction == ColSortedness::Descending)
            direction = ColSortedness::Ascending;
    }
    return {colSortedness};
}

std::vector<std::vector<ColSortedness>> FillOp::inferColSortedness() {
    auto matrix = dyn_cast<MatrixType>(getRes().getType());
    if (!matrix || matrix.getNumCols() < 0 || matrix.getElementType() != getArg().getType())
        return {{}};
    auto *constant = CompilerUtils::constantOfAnyType(getArg());
    if (!constant)
        return {{}};
    auto daphneConstant = dyn_cast<ConstantOp>(constant);
    auto value = daphneConstant ? daphneConstant.getValue() : cast<arith::ConstantOp>(constant).getValue();
    if (auto fp = dyn_cast<FloatAttr>(value)) {
        if (fp.getValue().isNaN())
            return {{}};
    } else if (!isa<IntegerAttr>(value))
        return {{}};
    // Equal values satisfy both directions. Use ascending for this inferred fact.
    return {std::vector<ColSortedness>(matrix.getNumCols(), ColSortedness::Ascending)};
}

std::vector<std::vector<ColSortedness>> mlir::daphne::tryInferColSortedness(Operation *op) {
    if (auto interface = dyn_cast<InferColSortedness>(op))
        return interface.inferColSortedness();
    return std::vector<std::vector<ColSortedness>>(op->getNumResults());
}
