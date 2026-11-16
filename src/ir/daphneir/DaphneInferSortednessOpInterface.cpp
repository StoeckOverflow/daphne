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
#include <ir/daphneir/DaphneInferSortednessOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<MatrixSortedness> daphne::OrderOp::inferSortedness() {
    auto matrix = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    auto [knownResult, returnIndices] = CompilerUtils::isConstant<bool>(getReturnIdxs());
    if (!matrix || matrix.getNumCols() != 1 || !knownResult || returnIndices || getColIdxs().size() != 1 ||
        getAscs().size() != 1)
        return {MatrixSortedness::Unknown};
    auto [knownKey, key] = CompilerUtils::isConstant<size_t>(getColIdxs()[0]);
    auto [knownDirection, ascending] = CompilerUtils::isConstant<bool>(getAscs()[0]);
    if (!knownKey || key != 0 || !knownDirection)
        return {MatrixSortedness::Unknown};
    return {ascending ? MatrixSortedness::SortedAsc : MatrixSortedness::SortedDesc};
}

std::vector<MatrixSortedness> daphne::EwLtOp::inferSortedness() {
    auto matrix = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    // Initially admit matching signed integers only: no NaNs or mixed-type conversions.
    if (!matrix || !matrix.getElementType().isSignedInteger(64) || getRhs().getType() != matrix.getElementType())
        return {MatrixSortedness::Unknown};

    auto [known, scalar] = CompilerUtils::isConstant<int64_t>(getRhs());
    if (known) {
        auto minimum = matrix.getMinValue();
        auto maximum = matrix.getMaxValue();
        const double value = static_cast<double>(scalar);
        if ((minimum && value <= *minimum) || (maximum && value > *maximum))
            return {MatrixSortedness::AllEqual};
    }

    // A fixed scalar threshold reverses direction, even if its value is dynamic.
    switch (matrix.getSortedness()) {
    case MatrixSortedness::SortedAsc:
        return {MatrixSortedness::SortedDesc};
    case MatrixSortedness::SortedDesc:
        return {MatrixSortedness::SortedAsc};
    case MatrixSortedness::AllEqual:
        return {MatrixSortedness::AllEqual};
    default:
        return {MatrixSortedness::Unknown};
    }
}

std::vector<MatrixSortedness> daphne::EwAddOp::inferSortedness() {
    auto lhs = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhs = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());
    if (!lhs || !rhs)
        return {MatrixSortedness::Unknown};

    // These rules assume order-preserving arithmetic.
    auto left = lhs.getSortedness();
    auto right = rhs.getSortedness();
    if (left == MatrixSortedness::AllEqual && right == MatrixSortedness::AllEqual)
        return {MatrixSortedness::AllEqual};
    if (left == MatrixSortedness::AllEqual)
        left = right;
    if (right == MatrixSortedness::AllEqual)
        right = left;
    if (left == right && (left == MatrixSortedness::SortedAsc || left == MatrixSortedness::SortedDesc))
        return {left};
    return {MatrixSortedness::Unknown};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<MatrixSortedness> daphne::tryInferSortedness(Operation *op) {
    if (auto inferSortednessOp = llvm::dyn_cast<daphne::InferSortedness>(op))
        // If the operation implements the inference interface, we apply that.
        return inferSortednessOp.inferSortedness();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<MatrixSortedness> sortednesses;
        for (size_t i = 0; i < op->getNumResults(); i++)
            sortednesses.push_back(MatrixSortedness::Unknown);
        return sortednesses;
    }
}
