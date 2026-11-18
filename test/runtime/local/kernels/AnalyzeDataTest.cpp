/*
 * Copyright 2026 The DAPHNE Consortium
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

#include <catch.hpp>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/AnalyzeData.h>
#include <tags.h>

using property_acquisition::BoundsAccumulator;
using property_acquisition::DistinctAccumulator;
using property_acquisition::NonzeroAccumulator;

namespace {
template <template <typename> class... Accumulators, class DT> void acquire(const DT *arg) {
    auto state = property_acquisition::AccumulatorSet(Accumulators<typename DT::VT>{}...);
    analyzeData(arg, state, nullptr);
}
} // namespace

TEMPLATE_PRODUCT_TEST_CASE("analyzeData observations and reacquisition", TAG_KERNELS, (DenseMatrix, CSRMatrix),
                           (int8_t, uint8_t, int32_t, uint32_t, int64_t, uint64_t, float, double)) {
    using DT = TestType;
    auto *matrix = genGivenVals<DT>(2, {0, 0, 2, 2, 0, 3}, 6);
    matrix->symmetric = BoolOrUnknown::True;
    acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
    CHECK(matrix->sparsity == 0.5);
    CHECK(matrix->lowerBound == 0);
    CHECK(matrix->upperBound == 3);
    CHECK(matrix->numDistinct == 3);
    matrix->set(0, 0, 4);
    acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
    CHECK(matrix->sparsity == Approx(4.0 / 6.0));
    CHECK(matrix->lowerBound == 0);
    CHECK(matrix->upperBound == 4);
    CHECK(matrix->numDistinct == 4);
    // Acquisition does not compute or overwrite symmetry.
    CHECK(matrix->symmetric == BoolOrUnknown::True);
    DataObjectFactory::destroy(matrix);
}

TEMPLATE_PRODUCT_TEST_CASE("analyzeData floating values and NaN recovery", TAG_KERNELS, (DenseMatrix, CSRMatrix),
                           (float, double)) {
    using DT = TestType;
    auto *matrix = genGivenVals<DT>(2, {0.0, -0.0, -INFINITY, INFINITY}, 4);
    acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
    CHECK(matrix->sparsity == 0.5);
    CHECK(matrix->numDistinct == 3);
    CHECK(matrix->lowerBound == -INFINITY);
    CHECK(matrix->upperBound == INFINITY);
    matrix->set(0, 0, std::numeric_limits<typename DT::VT>::quiet_NaN());
    acquire<NonzeroAccumulator>(matrix);
    CHECK(matrix->sparsity == 0.75);
    CHECK(matrix->numDistinct == 3);
    CHECK(matrix->lowerBound == -INFINITY);
    CHECK(matrix->upperBound == INFINITY);
    acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
    CHECK(matrix->sparsity == 0.75);
    CHECK(matrix->numDistinct == -1);
    CHECK_FALSE(matrix->lowerBound.has_value());
    CHECK_FALSE(matrix->upperBound.has_value());
    matrix->set(0, 0, 2);
    acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
    CHECK(matrix->sparsity == 0.75);
    CHECK(matrix->numDistinct == 4);
    CHECK(matrix->lowerBound == -INFINITY);
    CHECK(matrix->upperBound == INFINITY);
    DataObjectFactory::destroy(matrix);
}

TEST_CASE("analyzeData views", TAG_KERNELS) {
    SECTION("dense column slice respects row strides and backing-storage mutation") {
        using DT = DenseMatrix<double>;
        auto *matrix = genGivenVals<DT>(2, {99, 0, 2, 99, 99, 2, 0, 99});
        auto *view = DataObjectFactory::create<DT>(matrix, 0, 2, 1, 3);
        acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(view);
        CHECK(view->sparsity == 0.5);
        CHECK(view->lowerBound == 0);
        CHECK(view->upperBound == 2);
        CHECK(view->numDistinct == 2);
        matrix->set(0, 1, 5);
        acquire<BoundsAccumulator, DistinctAccumulator>(view);
        CHECK(view->lowerBound == 0);
        CHECK(view->upperBound == 5);
        CHECK(view->numDistinct == 3);
        CHECK(view->sparsity == 0.5);
        DataObjectFactory::destroy(view, matrix);
    }
    SECTION("CSR row slice excludes values outside the view") {
        using DT = CSRMatrix<double>;
        auto *matrix = genGivenVals<DT>(3, {0, 0, 9, 0, 2, 0, 3, 0, 0});
        auto *view = DataObjectFactory::create<DT>(matrix, 1, 3);
        acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(view);
        CHECK(view->sparsity == Approx(1.0 / 3.0));
        CHECK(view->lowerBound == 0);
        CHECK(view->upperBound == 3);
        CHECK(view->numDistinct == 3);
        DataObjectFactory::destroy(view, matrix);
    }
}

TEMPLATE_TEST_CASE("analyzeData empty matrices", TAG_KERNELS, DenseMatrix<double>, CSRMatrix<double>) {
    using DT = TestType;
    for (auto shape : {std::pair<size_t, size_t>{0, 0}, {0, 3}, {3, 0}}) {
        CAPTURE(shape.first, shape.second);
        DT *empty;
        if constexpr (std::is_same_v<DT, CSRMatrix<double>>)
            empty = DataObjectFactory::create<DT>(shape.first, shape.second, 0, true);
        else
            empty = DataObjectFactory::create<DT>(shape.first, shape.second, false);
        empty->lowerBound = -1;
        empty->upperBound = 1;
        acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(empty);
        CHECK(empty->sparsity == 0);
        CHECK(empty->numDistinct == 0);
        CHECK_FALSE(empty->lowerBound.has_value());
        CHECK_FALSE(empty->upperBound.has_value());
        DataObjectFactory::destroy(empty);
    }
}

TEST_CASE("analyzeData column bounds", TAG_KERNELS) {
    SECTION("integer extrema use the current double representation") {
        auto *column = DataObjectFactory::create<Column<int64_t>>(3, false);
        column->getValues()[0] = std::numeric_limits<int64_t>::min();
        column->getValues()[1] = 0;
        column->getValues()[2] = std::numeric_limits<int64_t>::max();
        acquire<BoundsAccumulator>(column);
        CHECK(column->lowerBound == static_cast<double>(std::numeric_limits<int64_t>::min()));
        CHECK(column->upperBound == static_cast<double>(std::numeric_limits<int64_t>::max()));
        DataObjectFactory::destroy(column);
    }
    SECTION("empty column clears old bounds") {
        auto *column = DataObjectFactory::create<Column<double>>(0, false);
        column->lowerBound = -1;
        column->upperBound = 1;
        acquire<BoundsAccumulator>(column);
        CHECK_FALSE(column->lowerBound.has_value());
        CHECK_FALSE(column->upperBound.has_value());
        DataObjectFactory::destroy(column);
    }
    SECTION("special floating values and recovery") {
        auto *column = DataObjectFactory::create<Column<double>>(3, false);
        column->getValues()[0] = -INFINITY;
        column->getValues()[1] = 0;
        column->getValues()[2] = INFINITY;
        acquire<BoundsAccumulator>(column);
        CHECK(column->lowerBound == -INFINITY);
        CHECK(column->upperBound == INFINITY);
        column->getValues()[1] = NAN;
        acquire<BoundsAccumulator>(column);
        CHECK_FALSE(column->lowerBound.has_value());
        CHECK_FALSE(column->upperBound.has_value());
        column->getValues()[1] = 2;
        acquire<BoundsAccumulator>(column);
        CHECK(column->lowerBound == -INFINITY);
        CHECK(column->upperBound == INFINITY);
        DataObjectFactory::destroy(column);
    }
    SECTION("empty accumulator set preserves bounds") {
        auto *column = DataObjectFactory::create<Column<double>>(1, true);
        column->lowerBound = -4;
        column->upperBound = 8;
        acquire<>(column);
        CHECK(column->lowerBound == -4);
        CHECK(column->upperBound == 8);
        DataObjectFactory::destroy(column);
    }
}

TEMPLATE_TEST_CASE("analyzeData accumulator selection", TAG_KERNELS, DenseMatrix<double>, CSRMatrix<double>) {
    auto *matrix = genGivenVals<TestType>(1, {0.0, -0.0, 2.0, 2.0});
    matrix->sparsity = -1;
    matrix->lowerBound = -7;
    matrix->upperBound = 7;
    matrix->numDistinct = -1;
    matrix->symmetric = BoolOrUnknown::True;
    bool sparsity = false, bounds = false, distinct = false;
    SECTION("none") { acquire<>(matrix); }
    SECTION("nonzeros") {
        acquire<NonzeroAccumulator>(matrix);
        sparsity = true;
    }
    SECTION("bounds") {
        acquire<BoundsAccumulator>(matrix);
        bounds = true;
    }
    SECTION("distinct") {
        acquire<DistinctAccumulator>(matrix);
        distinct = true;
    }
    SECTION("nonzeros and bounds") {
        acquire<NonzeroAccumulator, BoundsAccumulator>(matrix);
        sparsity = bounds = true;
    }
    SECTION("nonzeros and distinct") {
        acquire<NonzeroAccumulator, DistinctAccumulator>(matrix);
        sparsity = distinct = true;
    }
    SECTION("bounds and distinct") {
        acquire<BoundsAccumulator, DistinctAccumulator>(matrix);
        bounds = distinct = true;
    }
    SECTION("all") {
        acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
        sparsity = bounds = distinct = true;
    }
    CHECK(matrix->sparsity == (sparsity ? 0.5 : -1));
    CHECK(matrix->lowerBound == (bounds ? 0 : -7));
    CHECK(matrix->upperBound == (bounds ? 2 : 7));
    CHECK(matrix->numDistinct == (distinct ? 2 : -1));
    CHECK(matrix->symmetric == BoolOrUnknown::True);
    DataObjectFactory::destroy(matrix);
}

TEST_CASE("analyzeData CSR zero handling", TAG_KERNELS) {
    using DT = CSRMatrix<double>;
    SECTION("implicit zeros extend positive and negative bounds") {
        for (double sign : {-1.0, 1.0}) {
            auto *matrix = genGivenVals<DT>(1, {2 * sign, 0, 3 * sign});
            acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
            CHECK(matrix->sparsity == Approx(2.0 / 3.0));
            CHECK(matrix->lowerBound == (sign < 0 ? -3 : 0));
            CHECK(matrix->upperBound == (sign < 0 ? 0 : 3));
            CHECK(matrix->numDistinct == 3);
            DataObjectFactory::destroy(matrix);
        }
    }
    SECTION("explicit zeros with and without implicit zeros") {
        for (bool implicit : {false, true}) {
            auto *matrix = genGivenVals<DT>(1, {1.0, implicit ? 0.0 : 2.0, 3.0});
            matrix->getValues(0)[0] = 0;
            acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
            CHECK(matrix->sparsity == Approx((implicit ? 1.0 : 2.0) / 3.0));
            CHECK(matrix->lowerBound == 0);
            CHECK(matrix->upperBound == 3);
            CHECK(matrix->numDistinct == (implicit ? 2 : 3));
            DataObjectFactory::destroy(matrix);
        }
    }
    SECTION("nonempty all-zero matrix") {
        auto *matrix = genGivenVals<DT>(2, {0, 0, 0, 0});
        acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
        CHECK(matrix->sparsity == 0);
        CHECK(matrix->lowerBound == 0);
        CHECK(matrix->upperBound == 0);
        CHECK(matrix->numDistinct == 1);
        DataObjectFactory::destroy(matrix);
    }
    SECTION("fully stored matrix must not add zero") {
        auto *matrix = genGivenVals<DT>(2, {2, 3, 2, 3});
        acquire<NonzeroAccumulator, BoundsAccumulator, DistinctAccumulator>(matrix);
        CHECK(matrix->sparsity == 1);
        CHECK(matrix->lowerBound == 2);
        CHECK(matrix->upperBound == 3);
        CHECK(matrix->numDistinct == 2);
        DataObjectFactory::destroy(matrix);
    }
}

// These checks document lossy double storage, not exact or enclosing integer bounds.
TEMPLATE_TEST_CASE("analyzeData integer endpoint conversion", TAG_KERNELS, int64_t, uint64_t) {
    using VT = TestType;
    constexpr VT limit = VT(1) << 53;
    auto *matrix = genGivenVals<DenseMatrix<VT>>(1, {VT(0), limit});
    acquire<BoundsAccumulator>(matrix);
    CHECK(matrix->lowerBound == 0.0);
    CHECK(matrix->upperBound == static_cast<double>(limit));
    matrix->set(0, 1, limit + 1);
    acquire<BoundsAccumulator>(matrix);
    CHECK(matrix->lowerBound == 0.0);
    CHECK(matrix->upperBound == static_cast<double>(limit + 1));
    matrix->set(0, 0, std::numeric_limits<VT>::max());
    acquire<BoundsAccumulator>(matrix);
    CHECK(matrix->lowerBound == static_cast<double>(limit + 1));
    CHECK(matrix->upperBound == static_cast<double>(std::numeric_limits<VT>::max()));
    if constexpr (std::is_signed_v<VT>) {
        matrix->set(0, 0, -limit);
        matrix->set(0, 1, VT(0));
        acquire<BoundsAccumulator>(matrix);
        CHECK(matrix->lowerBound == static_cast<double>(-limit));
        CHECK(matrix->upperBound == 0.0);
        matrix->set(0, 0, -limit - 1);
        acquire<BoundsAccumulator>(matrix);
        CHECK(matrix->lowerBound == static_cast<double>(-limit - 1));
        CHECK(matrix->upperBound == 0.0);
    }
    DataObjectFactory::destroy(matrix);
}

TEST_CASE("analyzeData accumulator multiplicity", TAG_KERNELS) {
    property_acquisition::NonzeroAccumulator<double> nonzeros;
    property_acquisition::BoundsAccumulator<double> bounds;
    property_acquisition::DistinctAccumulator<double> distinct;
    auto update = [&](double value, size_t count) {
        nonzeros.update(value, count);
        bounds.update(value, count);
        distinct.update(value, count);
    };
    update(NAN, 0);
    update(2.0, 3);
    update(0.0, 5);
    update(-4.0, 0);
    CHECK(nonzeros.finalize(8) == Approx(3.0 / 8.0));
    const auto [lower, upper] = bounds.finalize();
    CHECK(lower == 0.0);
    CHECK(upper == 2.0);
    CHECK(distinct.finalize() == 2);
}
