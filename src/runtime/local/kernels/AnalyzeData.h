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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <util/DistinctSketch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace property_acquisition {

template <class Derived, typename VT> class AccumulatorBase {
  public:
    // count represents repeated occurrences, including implicit CSR zeros.
    void update(VT value, size_t count = 1) {
        if (count)
            static_cast<Derived *>(this)->updateValue(value, count);
    }

    template <class DT> void storeResult(DT *arg, size_t size) const {
        static_cast<const Derived *>(this)->writeResult(arg, size);
    }
};

template <typename VT> struct NonzeroAccumulator : AccumulatorBase<NonzeroAccumulator<VT>, VT> {
    size_t count = 0;

    void updateValue(VT value, size_t multiplicity) {
        if (value != VT(0))
            count += multiplicity;
    }
    double finalize(size_t size) const { return size ? double(count) / double(size) : 0.0; }
    template <class DT> void writeResult(DT *arg, size_t size) const { arg->sparsity = finalize(size); }
};

template <typename VT> struct BoundsAccumulator : AccumulatorBase<BoundsAccumulator<VT>, VT> {
    bool hasNaN = false;
    std::optional<VT> lower, upper;

    void updateValue(VT value, size_t) {
        if constexpr (std::is_floating_point_v<VT>) {
            if (std::isnan(value)) {
                hasNaN = true;
                return;
            }
        }
        lower = lower ? std::min(*lower, value) : value;
        upper = upper ? std::max(*upper, value) : value;
    }

    std::pair<std::optional<double>, std::optional<double>> finalize() const {
        if (hasNaN)
            return {};
        // TODO: Typed endpoints are needed to preserve full-range integer bounds.
        return {lower ? std::optional<double>(static_cast<double>(*lower)) : std::nullopt,
                upper ? std::optional<double>(static_cast<double>(*upper)) : std::nullopt};
    }
    template <class DT> void writeResult(DT *arg, size_t) const {
        auto [lo, hi] = finalize();
        arg->lowerBound = lo;
        arg->upperBound = hi;
    }
};

template <typename VT> struct DistinctAccumulator : AccumulatorBase<DistinctAccumulator<VT>, VT> {
    bool hasNaN = false;
    std::unordered_set<VT> values;

    void updateValue(VT value, size_t) {
        if constexpr (std::is_floating_point_v<VT>) {
            if (std::isnan(value)) {
                hasNaN = true;
                return;
            }
        }
        values.insert(value);
        if (values.size() > static_cast<size_t>(std::numeric_limits<ssize_t>::max()))
            throw std::overflow_error("analyzeData: distinct count overflows ssize_t");
    }
    ssize_t finalize() const { return hasNaN ? -1 : static_cast<ssize_t>(values.size()); }
    template <class DT> void writeResult(DT *arg, size_t) const {
        arg->numDistinct = finalize();
        arg->numDistinctEstimate.reset();
        arg->numDistinctLowerBound = -1;
    }
};

// Select one distinct-count accumulator per acquisition.
// Approximation affects distinct count only; other accumulators retain their contracts.
template <typename VT> struct ApproxDistinctAccumulator : AccumulatorBase<ApproxDistinctAccumulator<VT>, VT> {
    static constexpr size_t sketchSize = 64;
    static constexpr int64_t hashSeed = 1234567890;
    DistinctSketch<VT> sketch{sketchSize, hashSeed};
    bool hasNaN = false;

    void updateValue(VT value, size_t) {
        if constexpr (std::is_floating_point_v<VT>) {
            if (std::isnan(value)) {
                hasNaN = true;
                return;
            }
        }
        sketch.insert(value);
    }
    template <class DT> void writeResult(DT *arg, size_t size) const {
        arg->numDistinct = size == 0 ? 0 : -1;
        arg->numDistinctLowerBound = -1;
        arg->numDistinctEstimate = hasNaN ? std::nullopt : std::optional<double>(sketch.estimate(size));
    }
};

// Stop growing the set after the limit, but keep scanning for NaNs and other properties.
template <typename VT> struct BoundedDistinctAccumulator : AccumulatorBase<BoundedDistinctAccumulator<VT>, VT> {
    static constexpr size_t setLimit = 4096;
    std::unordered_set<VT> values;
    bool hasNaN = false;

    void updateValue(VT value, size_t) {
        if constexpr (std::is_floating_point_v<VT>) {
            if (std::isnan(value)) {
                hasNaN = true;
                return;
            }
        }
        if (values.size() <= setLimit)
            values.insert(value);
    }
    template <class DT> void writeResult(DT *arg, size_t) const {
        arg->numDistinct = -1;
        arg->numDistinctEstimate.reset();
        arg->numDistinctLowerBound = -1;
        if (!hasNaN) {
            if (values.size() > setLimit)
                arg->numDistinctLowerBound = static_cast<ssize_t>(values.size());
            else
                arg->numDistinct = static_cast<ssize_t>(values.size());
        }
    }
};

template <class... Accumulators> class AccumulatorSet {
    std::tuple<Accumulators...> accumulators;

  public:
    explicit AccumulatorSet(Accumulators... values) : accumulators(std::move(values)...) {}
    template <typename VT> void update(VT value, size_t count = 1) {
        std::apply([&](auto &...a) { (a.update(value, count), ...); }, accumulators);
    }
    template <class DT> void storeResults(DT *arg, size_t size) const {
        std::apply([&](const auto &...a) { (a.storeResult(arg, size), ...); }, accumulators);
    }
};

template <class... Accumulators> AccumulatorSet(Accumulators...) -> AccumulatorSet<Accumulators...>;

} // namespace property_acquisition

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DT> struct AnalyzeData {
    template <class Accumulators> static void apply(const DT *, Accumulators &, size_t, DCTX(ctx)) = delete;
};

// ----------------------------------------------------------------------------
// Convenience function
// ----------------------------------------------------------------------------

template <class DT, class... Accumulators>
void analyzeData(const DT *arg, property_acquisition::AccumulatorSet<Accumulators...> &state, DCTX(ctx)) {
    using VT = typename DT::VT;
    static_assert(std::is_same_v<VT, float> || std::is_same_v<VT, double> ||
                  (std::is_integral_v<VT> && !std::is_same_v<VT, bool> &&
                   (sizeof(VT) == 1 || sizeof(VT) == 4 || sizeof(VT) == 8)));
    if constexpr (sizeof...(Accumulators) > 0) {
        const size_t rows = arg->getNumRows(), cols = arg->getNumCols();
        if (cols && rows > std::numeric_limits<size_t>::max() / cols)
            throw std::overflow_error("analyzeData: element count overflows size_t");
        const size_t size = rows * cols;
        AnalyzeData<DT>::apply(arg, state, size, ctx);
        state.storeResults(const_cast<DT *>(arg), size);
    }
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct AnalyzeData<DenseMatrix<VT>> {
    template <class Accumulators>
    static void apply(const DenseMatrix<VT> *arg, Accumulators &state, size_t, DCTX(ctx)) {
        for (size_t r = 0; r < arg->getNumRows(); ++r) {
            const VT *values = arg->getNumCols() ? arg->getValues() + r * arg->getRowSkip() : nullptr;
            for (size_t c = 0; c < arg->getNumCols(); ++c)
                state.update(values[c]);
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct AnalyzeData<CSRMatrix<VT>> {
    template <class Accumulators>
    static void apply(const CSRMatrix<VT> *arg, Accumulators &state, size_t size, DCTX(ctx)) {
        size_t stored = 0;
        for (size_t r = 0; r < arg->getNumRows(); ++r) {
            const size_t count = arg->getNumNonZeros(r);
            const VT *values = arg->getValues(r);
            for (size_t c = 0; c < count; ++c)
                state.update(values[c]);
            stored += count;
        }
        if (stored < size)
            state.update(VT(0), size - stored);
    }
};

// ----------------------------------------------------------------------------
// Column
// ----------------------------------------------------------------------------

template <typename VT> struct AnalyzeData<Column<VT>> {
    template <class Accumulators> static void apply(const Column<VT> *arg, Accumulators &state, size_t, DCTX(ctx)) {
        const VT *values = arg->getValues();
        for (size_t r = 0; r < arg->getNumRows(); ++r)
            state.update(values[r]);
    }
};
