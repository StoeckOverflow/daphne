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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/AdaptiveCostModel.h>
#include <runtime/local/kernels/AnalyzeData.h>
#include <runtime/local/kernels/TransferProperties.h>

#include <catch.hpp>
#include <tags.h>

namespace {
struct AnalysisContext {
    DaphneUserConfig config;
    DaphneContext ctx{config, KernelDispatchMapping::instance(), Statistics::instance(), PropertyLogger::instance(),
                      StringRefCounter::instance()};
};

template <class DT>
void attach(const DT *arg, bool densityKnown, double density, bool minKnown, double min, bool maxKnown, double max,
            DaphneContext *ctx) {
    transferProperties(arg, densityKnown, density, false, -1, false, -1, minKnown, min, maxKnown, max, false, -1, false,
                       -1, ctx);
}
} // namespace

TEMPLATE_TEST_CASE("Property transfer attaches values without acquisition", TAG_KERNELS TAG_ADAPTIVE,
                   DenseMatrix<double>, CSRMatrix<double>) {
    auto *arg = genGivenVals<TestType>(2, {0., 2., 0., 3.});
    AnalysisContext context;
    context.config.automaticallyAnalyzeEverything = true;

    // Deliberately different from the actual density and bounds: transfer must not scan.
    attach(arg, true, 0.875, true, -1.0, true, 2.5, &context.ctx);
    CHECK(arg->is_sparsity);
    CHECK(arg->sparsity == 0.875);
    CHECK(arg->is_minValue);
    CHECK(arg->minValue == -1.0);
    CHECK(arg->is_maxValue);
    CHECK(arg->maxValue == 2.5);
    CHECK_FALSE(arg->is_distinct);
    CHECK_FALSE(arg->is_symmetric);
    CHECK_FALSE(arg->is_sortness);

    attach(arg, false, 0., false, 0., false, 0., &context.ctx);
    CHECK(arg->sparsity == 0.875);
    CHECK(arg->minValue == -1.0);
    CHECK(arg->maxValue == 2.5);
    DataObjectFactory::destroy(arg);
}

TEST_CASE("Property acquisition is selective and preserves known values", TAG_KERNELS TAG_ADAPTIVE) {
    auto *arg = genGivenVals<DenseMatrix<double>>(2, {0., 2., 0., 3.});
    AnalysisContext context;
    // The runtime flag alone must not broaden an explicit request.
    context.config.automaticallyAnalyzeEverything = true;
    analyzeData(arg, true, false, false, false, false, false, &context.ctx);
    CHECK(arg->is_sparsity);
    CHECK(arg->sparsity == 0.5);
    CHECK_FALSE(arg->is_minValue);
    CHECK_FALSE(arg->is_maxValue);
    CHECK_FALSE(arg->is_distinct);
    CHECK_FALSE(arg->is_symmetric);
    CHECK_FALSE(arg->is_sortness);

    attach(arg, true, 0.875, true, -17., false, 0., &context.ctx);
    analyzeData(arg, true, false, false, true, false, false, &context.ctx);
    CHECK(arg->sparsity == 0.875);
    CHECK(arg->minValue == -17.);
    CHECK(arg->is_maxValue);
    CHECK(arg->maxValue == 3.);

    analyzeData(arg, true, false, false, true, false, false, &context.ctx);
    CHECK(arg->sparsity == 0.875);
    CHECK(arg->minValue == -17.);
    CHECK(arg->maxValue == 3.);
    DataObjectFactory::destroy(arg);
}

TEST_CASE("Column property transfer and partial bound acquisition", TAG_KERNELS TAG_ADAPTIVE) {
    auto *arg = genGivenVals<Column<int64_t>>(4, {-2, 1, 5, 3});
    AnalysisContext context;
    context.config.automaticallyAnalyzeEverything = true;
    attach(arg, false, 0., true, -1., false, 0., &context.ctx);
    CHECK(arg->is_minValue);
    CHECK(arg->minValue == -1.);
    CHECK_FALSE(arg->is_maxValue);

    analyzeData(arg, false, false, false, true, false, false, &context.ctx);
    CHECK(arg->minValue == -1.);
    CHECK(arg->is_maxValue);
    CHECK(arg->maxValue == 5.);
    attach(arg, false, 0., false, 0., false, 0., &context.ctx);
    CHECK(arg->minValue == -1.);
    CHECK(arg->maxValue == 5.);
    DataObjectFactory::destroy(arg);
}

TEST_CASE("CSR property acquisition retains unsupported requests as unknown", TAG_KERNELS TAG_ADAPTIVE) {
    auto *arg = genGivenVals<CSRMatrix<double>>(2, {0., 2., 0., 3.});
    AnalysisContext context;
    const bool patternKnown = arg->is_sparsityPatternID;
    const auto patternID = arg->sparsityPatternID;
    attach(arg, true, 0.5, false, 0., false, 0., &context.ctx);
    analyzeData(arg, true, true, true, true, true, true, &context.ctx);
    CHECK(arg->is_sparsity);
    CHECK(arg->sparsity == 0.5);
    CHECK_FALSE(arg->is_minValue);
    CHECK_FALSE(arg->is_maxValue);
    CHECK_FALSE(arg->is_distinct);
    CHECK_FALSE(arg->is_symmetric);
    CHECK_FALSE(arg->is_sortness);
    CHECK(arg->is_sparsityPatternID == patternKnown);
    CHECK(arg->sparsityPatternID == patternID);
    DataObjectFactory::destroy(arg);
}

TEST_CASE("Distinct acquisition preserves exact and early-abort modes", TAG_KERNELS TAG_ADAPTIVE) {
    using Mode = DaphneUserConfig::AdaptiveAnalyzeMode;
    for (auto mode : {Mode::Exact, Mode::EarlyAbort}) {
        DYNAMIC_SECTION("mode " << static_cast<int>(mode)) {
            auto *arg = genGivenVals<DenseMatrix<double>>(2, {0., 2., 0., 3.});
            AnalysisContext context;
            context.config.adaptiveAnalyze = true;
            context.config.adaptiveAnalyzeMode = mode;
            context.config.mapDistinctThresholdIsRelative = false;
            context.config.mapDistinctThresholdAbsolute = 1;
            analyzeData(arg, false, false, false, false, true, false, &context.ctx);
            CHECK(arg->is_distinct);
            CHECK(arg->distinct == (mode == Mode::Exact ? 3 : 2));
            CHECK_FALSE(arg->is_sparsity);
            DataObjectFactory::destroy(arg);
        }
    }
}

TEST_CASE("Map cost model compares remaining execution costs", TAG_KERNELS TAG_ADAPTIVE) {
    const AdaptiveCostModel::CostParams costs(100., 10., 20.);
    const auto cached = AdaptiveCostModel::analyzeCosts(70, 100, costs);
    CHECK(cached.defaultCost == 10000.);
    CHECK(cached.cacheCost == 9400.);
    CHECK(cached.savings == 600.);
    CHECK(cached.recommendCache);
    CHECK(cached.threshold == 75);
    CHECK(AdaptiveCostModel::calculateThresholdRatio(100, costs) == Approx(0.75));
    CHECK_FALSE(AdaptiveCostModel::analyzeCosts(80, 100, costs).recommendCache);
}
