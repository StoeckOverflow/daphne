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

#include <api/cli/DaphneUserConfig.h>
#include <compiler/utils/CompilerUtils.h>

#include <llvm/ADT/DenseSet.h>
#include <mlir/Pass/Pass.h>


using namespace mlir;

namespace {
void emitAnalysis(OpBuilder &builder, Location loc, Value input, daphne::AnalysisProperty properties) {
    builder.create<daphne::AnalyzeDataOp>(loc, input, properties);
}

void emitBoundsTransfer(OpBuilder &builder, Location loc, Value input, Value min, Value max) {
    auto yes = builder.create<daphne::ConstantOp>(loc, true);
    auto no = builder.create<daphne::ConstantOp>(loc, false);
    auto unusedFloat = builder.create<daphne::ConstantOp>(loc, -1.0);
    auto unusedInteger = builder.create<daphne::ConstantOp>(loc, int64_t(-1));
    builder.create<daphne::TransferPropertiesOp>(loc, input,
        no, unusedFloat,   // density
        no, unusedInteger, // symmetry
        no, unusedInteger, // ordering
        yes, min, yes, max,
        no, unusedInteger, // distinct count
        no, unusedInteger  // sparsity pattern
    );
}
} // namespace

struct AdaptiveAnalyzePropertiesPass : public PassWrapper<AdaptiveAnalyzePropertiesPass, OperationPass<func::FuncOp>> {
    explicit AdaptiveAnalyzePropertiesPass(const DaphneUserConfig &cfg) : userConfig(cfg) {}

    void runOnOperation() final;
    StringRef getArgument() const final { return "adaptive-analyze-props"; }
    StringRef getDescription() const final {
        return "Insert explicit automatic or consumer-requested property analysis";
    }

    void processBlock(OpBuilder &builder, Block *b, llvm::DenseSet<Value> automaticallyAnalyzed = {});
    void acquireBounds(OpBuilder &builder, Location loc, Value input);

    const DaphneUserConfig &userConfig;
};

void AdaptiveAnalyzePropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();

    if (!userConfig.adaptiveAnalyze && !userConfig.automaticallyAnalyzeEverything)
        return;

    OpBuilder builder(f.getContext());
    for (Block &block : f.getBody())
        processBlock(builder, &block);
}

void AdaptiveAnalyzePropertiesPass::processBlock(OpBuilder &builder, Block *b,
                                                  llvm::DenseSet<Value> automaticallyAnalyzed) {
    for (Operation &op : llvm::make_early_inc_range(*b)) {
        // Static transfers already follow matrix producers. Acquire missing properties
        // after attaching those known values, at the same points as the former runtime auto mode.
        if (userConfig.automaticallyAnalyzeEverything) {
            if (auto transfer = dyn_cast<daphne::TransferPropertiesOp>(&op)) {
                builder.setInsertionPointAfter(&op);
                emitAnalysis(builder, op.getLoc(), transfer.getArg(),
                             daphne::AnalysisProperty::All);
                automaticallyAnalyzed.insert(transfer.getArg());
            }
        }
        auto exploitation = dyn_cast<daphne::PropertyExploitationOpInterface>(&op);
        if (userConfig.adaptiveAnalyze && exploitation) {
            SmallVector<daphne::PropertyOpportunity> opportunities;
            exploitation.getPropertyOpportunities(opportunities);
            for (const auto &opportunity : opportunities) {
                Value input = op.getOperand(opportunity.operandIndex);
                if (automaticallyAnalyzed.contains(input))
                    continue;
                builder.setInsertionPoint(&op);
                auto remaining = opportunity.properties;
                auto mt = ::mlir::dyn_cast<daphne::MatrixType>(input.getType());
                if (userConfig.use_vectorized_restricted && daphne::bitEnumContainsAll(remaining, daphne::AnalysisProperty::MinMax) && mt) {
                    if (!mt.getMinValue().has_value() || !mt.getMaxValue().has_value())
                        acquireBounds(builder, op.getLoc(), input);
                    remaining = remaining & ~daphne::AnalysisProperty::MinMax;
                }
                if (remaining != daphne::AnalysisProperty::None)
                    emitAnalysis(builder, op.getLoc(), input, remaining);
            }
        }
        for (Region &region : op.getRegions())
            for (Block &block : region.getBlocks())
                processBlock(builder, &block, automaticallyAnalyzed);
    }
}

void AdaptiveAnalyzePropertiesPass::acquireBounds(OpBuilder &builder, Location loc, Value input) {
    Type scalarTy = ::mlir::cast<daphne::MatrixType>(input.getType()).getElementType();
    Value min, max;
    if (userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd) {
        min = builder.create<daphne::MinAllSimdOp>(loc, scalarTy, input);
        max = builder.create<daphne::MaxAllSimdOp>(loc, scalarTy, input);
    } else {
        min = builder.create<daphne::AllAggMinOp>(loc, scalarTy, input);
        max = builder.create<daphne::AllAggMaxOp>(loc, scalarTy, input);
    }
    for (Value bound : {min, max})
        bound.getDefiningOp()->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
    min = builder.create<daphne::CastOp>(loc, builder.getF64Type(), min);
    max = builder.create<daphne::CastOp>(loc, builder.getF64Type(), max);
    emitBoundsTransfer(builder, loc, input, min, max);
}

std::unique_ptr<Pass> daphne::createAdaptiveAnalyzePropertiesPass(const DaphneUserConfig &cfg) {
    return std::make_unique<AdaptiveAnalyzePropertiesPass>(cfg);
}

// Version for auto-generated code (Passes.td)
std::unique_ptr<Pass> daphne::createAdaptiveAnalyzePropertiesPass() {
    static DaphneUserConfig defaultConfig;
    return std::make_unique<AdaptiveAnalyzePropertiesPass>(defaultConfig);
}
