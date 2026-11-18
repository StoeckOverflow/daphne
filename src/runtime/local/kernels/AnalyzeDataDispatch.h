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

#pragma once

#include <runtime/local/kernels/AnalyzeData.h>
#include <runtime/local/kernels/IsSymmetric.h>

// Kernel-catalog entry point: select a precompiled accumulator combination from runtime requests.
template <class DT>
void analyzeData(const DT *arg, bool sparsity, bool symmetric, bool bounds, bool distinct, DCTX(ctx)) {
    using VT = typename DT::VT;
    using namespace property_acquisition;
    if (!(sparsity || symmetric || bounds || distinct))
        return;
    if (!ctx)
        throw std::invalid_argument("analyzeData: acquisition requires a DaphneContext");
    const auto mode = ctx->getUserConfig().acquisitionMode;
    if (mode != AcquisitionMode::Exact && mode != AcquisitionMode::Approx && mode != AcquisitionMode::EarlyAbort)
        throw std::invalid_argument("analyzeData: invalid acquisition mode");

    auto runAcquisition = [&](auto... accumulators) {
        auto state = AccumulatorSet(std::move(accumulators)...);
        analyzeData(arg, state, ctx);
    };
    if constexpr (std::is_same_v<DT, Column<VT>>) {
        if (sparsity || symmetric || distinct)
            throw std::invalid_argument("analyzeData: columns support bounds acquisition only");
        runAcquisition(BoundsAccumulator<VT>{});
    } else {
        static_assert(std::is_same_v<DT, DenseMatrix<VT>> || std::is_same_v<DT, CSRMatrix<VT>>);
        auto selectDistinctAccumulator = [&](auto... accumulators) {
            if (!distinct) {
                runAcquisition(std::move(accumulators)...);
                return;
            }
            switch (mode) {
            case AcquisitionMode::Exact:
                runAcquisition(std::move(accumulators)..., DistinctAccumulator<VT>{});
                break;
            case AcquisitionMode::Approx:
                runAcquisition(std::move(accumulators)..., ApproxDistinctAccumulator<VT>{});
                break;
            case AcquisitionMode::EarlyAbort:
                runAcquisition(std::move(accumulators)..., BoundedDistinctAccumulator<VT>{});
                break;
            }
        };
        auto selectBoundsAccumulator = [&](auto... accumulators) {
            if (bounds)
                selectDistinctAccumulator(std::move(accumulators)..., BoundsAccumulator<VT>{});
            else
                selectDistinctAccumulator(std::move(accumulators)...);
        };
        if (sparsity)
            selectBoundsAccumulator(NonzeroAccumulator<VT>{});
        else
            selectBoundsAccumulator();
        if (symmetric)
            const_cast<DT *>(arg)->symmetric = isSymmetric(arg, ctx) ? BoolOrUnknown::True : BoolOrUnknown::False;
    }
}
