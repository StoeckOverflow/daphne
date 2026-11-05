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

#include <mlir/Pass/Pass.h>

#include <algorithm>
#include <set>

using namespace mlir;

namespace {
enum class Property { Sparsity, Symmetry, Sortedness, MinMax, Distinct, SparsityPattern };
using PropertySet = std::set<Property>;

PropertySet parseProperties(const std::vector<std::string> &names) {
    PropertySet properties;
    auto add = [&](Property property, std::initializer_list<const char *> aliases) {
        if (std::any_of(aliases.begin(), aliases.end(), [&](const char *name) {
                return std::find(names.begin(), names.end(), name) != names.end();
            }))
            properties.insert(property);
    };
    add(Property::Sparsity, {"sparsity"});
    add(Property::Symmetry, {"symmetric"});
    add(Property::Sortedness, {"sortness", "sorted"});
    add(Property::MinMax, {"minmax", "min", "max"});
    add(Property::Distinct, {"distinct", "numDistinct", "distinctCount"});
    add(Property::SparsityPattern, {"sparsityPattern"});
    return properties;
}

void emitAnalysis(OpBuilder &builder, Location loc, Value input, const PropertySet &properties) {
    // Adapt the typed request to the existing operation/kernel ABI.
    auto flag = [&](Property property) {
        return builder.create<daphne::ConstantOp>(loc, properties.contains(property));
    };
    builder.create<daphne::AnalyzeDataOp>(loc, input, flag(Property::Sparsity), flag(Property::Symmetry),
                                        flag(Property::Sortedness), flag(Property::MinMax), flag(Property::Distinct),
                                        flag(Property::SparsityPattern));
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

// Plan acquisition before context insertion; lowering supplies runtime arguments.
struct AdaptiveAnalyzePropertiesPass : public PassWrapper<AdaptiveAnalyzePropertiesPass, OperationPass<func::FuncOp>> {
    explicit AdaptiveAnalyzePropertiesPass(const DaphneUserConfig &cfg) : userConfig(cfg) {}

    void runOnOperation() final;
    StringRef getArgument() const final { return "adaptive-analyze-props"; }
    StringRef getDescription() const final {
        return "Insert analyzeData calls for adaptive property analysis based on adaptive_map";
    }

    void processBlock(OpBuilder &builder, Block *b);
    void acquireBounds(OpBuilder &builder, Location loc, Value input);

    const DaphneUserConfig &userConfig;
};

void AdaptiveAnalyzePropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();

    if (!userConfig.adaptiveAnalyze)
        return;

    OpBuilder builder(f.getContext());
    for (Block &block : f.getBody())
        processBlock(builder, &block);
}

void AdaptiveAnalyzePropertiesPass::processBlock(OpBuilder &builder, Block *b) {
    for (Operation &op : llvm::make_early_inc_range(*b)) {
        auto it = userConfig.adaptive_map.find(op.getName().stripDialect().str());
        if (it != userConfig.adaptive_map.end()) {
            const PropertySet request = parseProperties(it->second);
            SmallVector<Value> inputsToAnalyze;
            for (Value input : op.getOperands())
                if (input.getType().isa<daphne::MatrixType, daphne::FrameType, daphne::ColumnType>())
                    inputsToAnalyze.push_back(input);

            // Each consumer gets a local request; do not reuse across branches.
            for (Value input : inputsToAnalyze) {
                builder.setInsertionPoint(&op);
                auto remaining = request;
                auto mt = input.getType().dyn_cast<daphne::MatrixType>();
                if (userConfig.use_vectorized_restricted && remaining.contains(Property::MinMax) && mt) {
                    if (!mt.getMinValue().has_value() || !mt.getMaxValue().has_value())
                        acquireBounds(builder, op.getLoc(), input);
                    remaining.erase(Property::MinMax);
                }
                if (!remaining.empty())
                    emitAnalysis(builder, op.getLoc(), input, remaining);
            }
        }
        for (Region &region : op.getRegions())
            for (Block &block : region.getBlocks())
                processBlock(builder, &block);
    }
}

void AdaptiveAnalyzePropertiesPass::acquireBounds(OpBuilder &builder, Location loc, Value input) {
    Type scalarTy = input.getType().cast<daphne::MatrixType>().getElementType();
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
