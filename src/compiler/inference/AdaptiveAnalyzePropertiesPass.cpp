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
#include <ir/daphneir/Passes.h>

#include <ir/daphneir/DaphnePropertyUsageOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <array>

namespace mlir {
#define GEN_PASS_DECL_ADAPTIVEANALYZEPROPERTIES
#define GEN_PASS_DEF_ADAPTIVEANALYZEPROPERTIES
#include "ir/daphneir/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
enum RequestIndex { NonzeroCount, NumericalSymmetry, ValueBounds, DistinctCount, NumRequests };
using Requests = std::array<bool, NumRequests>;
struct AdaptiveAnalyzePropertiesPass : impl::AdaptiveAnalyzePropertiesBase<AdaptiveAnalyzePropertiesPass> {
    AdaptiveAnalyzePropertiesPass() = default;
    const KernelCatalog *catalog = nullptr;
    explicit AdaptiveAnalyzePropertiesPass(const KernelCatalog *catalog) : catalog(catalog) {}

    // Remove unsupported requests and return whether any supported request remains.
    static bool filterSupportedRequests(Value input, Requests &requests) {
        Type type = input.getType();
        if (!isa<daphne::MatrixType, daphne::ColumnType>(type))
            return false;
        Type element = CompilerUtils::getValueType(type);
        if (!(element.isF32() || element.isF64() || element.isSignedInteger(8) || element.isUnsignedInteger(8) ||
              element.isSignedInteger(32) || element.isUnsignedInteger(32) || element.isSignedInteger(64) ||
              element.isUnsignedInteger(64)))
            return false;
        if (isa<daphne::ColumnType>(type))
            requests[NonzeroCount] = requests[NumericalSymmetry] = requests[DistinctCount] = false;
        return llvm::is_contained(requests, true);
    }

    // Check whether this acquisition requests all needed properties of the same input.
    static bool coversRequests(daphne::AnalyzeDataOp analysis, Value input, const Requests &requests) {
        if (analysis.getArg() != input)
            return false;
        for (unsigned i = 0; i < requests.size(); ++i) {
            if (!requests[i])
                continue;
            auto [known, enabled] = CompilerUtils::isConstant<bool>(analysis.getOperands().drop_front()[i]);
            if (!known || !enabled)
                return false;
        }
        return true;
    }

    // Check the supplied compile-time catalog for a matching runtime acquisition kernel.
    bool hasAcquisitionKernel(Value input) const {
        // Standalone daphne-opt runs validate insertion without checking kernel availability.
        if (!catalog)
            return true;
        auto *context = input.getContext();
        std::vector<Type> args{KernelCatalog::normalizeTypeForKernelLookup(input.getType(), false)};
        args.insert(args.end(), NumRequests, IntegerType::get(context, 1));
        return catalog->findKernel("analyzeData", args, {}).has_value();
    }

    // Find a covering preceding acquisition, stopping at anything other than constants or acquisitions.
    bool hasLocalAcquisition(Operation &consumer, Value input, const Requests &requests) {
        for (Operation *prev = consumer.getPrevNode(); prev; prev = prev->getPrevNode()) {
            if (isa<daphne::ConstantOp, arith::ConstantOp>(prev))
                continue;
            auto analysis = dyn_cast<daphne::AnalyzeDataOp>(prev);
            if (!analysis)
                break;
            if (coversRequests(analysis, input, requests))
                return true;
        }
        return false;
    }

    void processBlock(Block &block) {
        OpBuilder builder(block.getParentOp()->getContext());
        for (Operation &op : llvm::make_early_inc_range(block)) {
            if (isa<daphne::AnalyzeDataOp>(op))
                continue;
            auto interface = dyn_cast<daphne::PropertyUsage>(&op);
            if (interface) {
                llvm::SmallVector<std::pair<Value, Requests>> inputs;
                for (unsigned operandIndex = 0; operandIndex < op.getNumOperands(); ++operandIndex) {
                    llvm::SmallVector<daphne::DataPropertyKind> properties;
                    interface.getUsedProperties(operandIndex, properties);
                    if (properties.empty())
                        continue;
                    Value input = op.getOperand(operandIndex);
                    auto it = llvm::find_if(inputs, [&](const auto &entry) { return entry.first == input; });
                    if (it == inputs.end()) {
                        inputs.emplace_back(input, Requests{});
                        it = std::prev(inputs.end());
                    }
                    for (auto property : properties) {
                        switch (property) {
                        case daphne::DataPropertyKind::NonzeroCount:
                            it->second[NonzeroCount] = true;
                            break;
                        case daphne::DataPropertyKind::NumericalSymmetry:
                            it->second[NumericalSymmetry] = true;
                            break;
                        case daphne::DataPropertyKind::ValueBounds:
                            it->second[ValueBounds] = true;
                            break;
                        case daphne::DataPropertyKind::DistinctCount:
                            it->second[DistinctCount] = true;
                            break;
                        }
                    }
                }
                for (auto &[input, requests] : inputs) {
                    if (!filterSupportedRequests(input, requests) || !hasAcquisitionKernel(input) ||
                        hasLocalAcquisition(op, input, requests))
                        continue;
                    builder.setInsertionPoint(&op);
                    llvm::SmallVector<Value> flags;
                    for (bool request : requests)
                        flags.push_back(builder.create<daphne::ConstantOp>(op.getLoc(), request));
                    builder.create<daphne::AnalyzeDataOp>(op.getLoc(), input, flags[NonzeroCount],
                                                          flags[NumericalSymmetry], flags[ValueBounds],
                                                          flags[DistinctCount]);
                }
            }

            if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op))
                for (Region &region : op.getRegions())
                    for (Block &nested : region)
                        processBlock(nested);
        }
    }

    void runOnOperation() override {
        for (Block &block : getOperation().getBody())
            processBlock(block);
    }
};
} // namespace

std::unique_ptr<Pass> daphne::createAdaptiveAnalyzePropertiesPass(const KernelCatalog *catalog) {
    return std::make_unique<AdaptiveAnalyzePropertiesPass>(catalog);
}
