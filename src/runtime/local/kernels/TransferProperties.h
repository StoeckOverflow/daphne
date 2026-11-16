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

#include <ir/daphneir/DataPropertyTypes.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DT> struct TransferProperties {
    static void apply(const DT *arg, double sparsity, int64_t symmetric, int64_t numDistinct, bool hasLowerBound,
                      double lowerBound, bool hasUpperBound, double upperBound, const int64_t *colSortedness,
                      size_t numColSortedness, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT>
void transferProperties(const DT *arg, double sparsity, int64_t symmetric, int64_t numDistinct, bool hasLowerBound,
                        double lowerBound, bool hasUpperBound, double upperBound, const int64_t *colSortedness,
                        size_t numColSortedness, DCTX(ctx)) {
    TransferProperties<DT>::apply(arg, sparsity, symmetric, numDistinct, hasLowerBound, lowerBound, hasUpperBound,
                                  upperBound, colSortedness, numColSortedness, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct TransferProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, double sparsity, int64_t symmetric, int64_t numDistinct,
                      bool hasLowerBound, double lowerBound, bool hasUpperBound, double upperBound,
                      const int64_t *colSortedness, size_t numColSortedness, DCTX(ctx)) {
        auto *matrix = const_cast<DenseMatrix<VT> *>(arg);
        matrix->colSortedness.resize(numColSortedness);
        for (size_t c = 0; c < numColSortedness; ++c)
            matrix->colSortedness[c] = static_cast<ColSortedness>(colSortedness[c]);
        matrix->sparsity = sparsity;
        matrix->symmetric = static_cast<BoolOrUnknown>(symmetric);
        matrix->numDistinct = numDistinct;
        matrix->lowerBound = hasLowerBound ? std::optional<double>(lowerBound) : std::nullopt;
        matrix->upperBound = hasUpperBound ? std::optional<double>(upperBound) : std::nullopt;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct TransferProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, double sparsity, int64_t symmetric, int64_t numDistinct,
                      bool hasLowerBound, double lowerBound, bool hasUpperBound, double upperBound,
                      const int64_t *colSortedness, size_t numColSortedness, DCTX(ctx)) {
        auto *matrix = const_cast<CSRMatrix<VT> *>(arg);
        matrix->colSortedness.resize(numColSortedness);
        for (size_t c = 0; c < numColSortedness; ++c)
            matrix->colSortedness[c] = static_cast<ColSortedness>(colSortedness[c]);
        matrix->sparsity = sparsity;
        matrix->symmetric = static_cast<BoolOrUnknown>(symmetric);
        matrix->numDistinct = numDistinct;
        matrix->lowerBound = hasLowerBound ? std::optional<double>(lowerBound) : std::nullopt;
        matrix->upperBound = hasUpperBound ? std::optional<double>(upperBound) : std::nullopt;
    }
};
