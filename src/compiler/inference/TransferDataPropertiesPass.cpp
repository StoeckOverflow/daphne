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
#include <ir/daphneir/Passes.h>

#include <mlir/Pass/Pass.h>

using namespace mlir;

/**
 * @brief Inserts `TransferPropertiesOp`s into the IR to ensure that compile-time information on the data properties of
 * DAPHNE data objects is transferred to the DAPHNE runtime.
 *
 * For each matrix-typed result of an operation, a `TransferPropertiesOp` is inserted that passes the compile-time
 * information on the matrix's data properties (extracted from the `MatrixType`) to the runtime. By inserting these
 * operations right after the creation of matrix-typed op results, we ensure that the runtime kernels subsequent
 * operations are lowered to have access to the property information. With this information available, the kernels can
 * do various runtime optimizations, e.g., choosing a more efficient algorithm on sorted or symmetric data.
 */
struct TransferDataPropertiesPass : public PassWrapper<TransferDataPropertiesPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;
    StringRef getArgument() const final { return "transfer-data-props"; }
    StringRef getDescription() const final { return "TODO"; }

    void processBlock(OpBuilder builder, Block *b);
};

void TransferDataPropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());
    processBlock(builder, &(f.getBody().front()));
}

void TransferDataPropertiesPass::processBlock(OpBuilder builder, Block *b) {
    for (Operation &op : b->getOperations()) {
        Location loc = op.getLoc();

        // Transfer known compiler properties to runtime objects.
        builder.setInsertionPointAfter(&op);

        for (Value v : op.getResults()) {
            Type t = v.getType();
            auto mt = t.dyn_cast<daphne::MatrixType>();
            if (!mt)
                continue;

            bool isSparsity = mt.getSparsity() != -1.0;
            auto coIsSparsity = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSparsity));
            auto coSparsity = builder.create<daphne::ConstantOp>(loc, isSparsity ? mt.getSparsity() : -1.0);

            bool isSymmetric = mt.getSymmetric() != BoolOrUnknown::Unknown;
            auto coIsSymmetric = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSymmetric));
            auto coSymmetric = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSymmetric()));

            bool isSortness = mt.getSortness() != MatrixSortness::Unknown;
            auto coIsSortness = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSortness));
            auto coSortness = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSortness()));

            Value coIsMinValue;
            Value coMinValue;
            if (mt.getMinValue().has_value()) {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMinValue = builder.create<daphne::ConstantOp>(loc, static_cast<double>(mt.getMinValue().value()));
            } else {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
                coMinValue = builder.create<daphne::ConstantOp>(loc, -1.0);
            }

            Value coIsMaxValue;
            Value coMaxValue;
            if (mt.getMaxValue().has_value()) {
                coIsMaxValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMaxValue = builder.create<daphne::ConstantOp>(loc, static_cast<double>(mt.getMaxValue().value()));
            } else {
                coIsMaxValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
                coMaxValue = builder.create<daphne::ConstantOp>(loc, -1.0);
            }

            bool isDistinct = mt.getDistinct() != -1;
            auto coIsDistinct = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isDistinct));
            auto coDistinct = builder.create<daphne::ConstantOp>(loc, isDistinct ? mt.getDistinct() : -1);

            bool isSparsityPatternID = mt.getSparsityPatternID() != -1;
            auto coIsSparsityPatternID = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSparsityPatternID));
            auto coSparsityPatternID = builder.create<daphne::ConstantOp>(loc,
                                                                           isSparsityPatternID ? mt.getSparsityPatternID() : -1);

            builder.create<daphne::TransferPropertiesOp>(loc, v, coIsSparsity, coSparsity, coIsSymmetric, coSymmetric,
                                                         coIsSortness, coSortness, coIsMinValue, coMinValue, coIsMaxValue,
                                                         coMaxValue, coIsDistinct, coDistinct, coIsSparsityPatternID,
                                                         coSparsityPatternID);
        }

        // Recurse into the op, if it has regions.
        for (Region &r : op.getRegions())
            for (Block &b2 : r.getBlocks())
                processBlock(builder, &b2);
    }
}

std::unique_ptr<Pass> daphne::createTransferDataPropertiesPass() {
    return std::make_unique<TransferDataPropertiesPass>();
}
