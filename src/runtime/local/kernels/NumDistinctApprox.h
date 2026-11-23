/*
 * Copyright 2021 The DAPHNE Consortium
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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <util/KMVSketch.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct NumDistinctApprox {
    static double apply(const DTArg *arg, size_t K, int64_t seed, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Estimates the number of distinct numeric values using a 64-bit KMV sketch.
 *
 * Requires K >= 2. Empty matrices return zero.
 * Signed zeros count as one value. NaNs are rejected, infinities are supported.
 * A fixed seed is reproducible. Seed -1 selects a time-based seed.
 */
template <class DTArg> double numDistinctApprox(const DTArg *arg, size_t K, int64_t seed, DCTX(ctx)) {
    return NumDistinctApprox<DTArg>::apply(arg, K, seed, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different DataTypes
// ****************************************************************************

// ----------------------------------------------------------------------------
// double <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct NumDistinctApprox<DenseMatrix<VT>> {
    static double apply(const DenseMatrix<VT> *arg, size_t K, int64_t seed, DCTX(ctx)) {
        KMVSketch<VT> sketch(K, seed);
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numRows == 0 || numCols == 0)
            return sketch.estimate();
        const VT *values = arg->getValues();
        const size_t rowSkip = arg->getRowSkip();
        for (auto rowIdx = 0ul; rowIdx < numRows; ++rowIdx)
            for (size_t colIdx = 0; colIdx < numCols; ++colIdx)
                sketch.insert(values[rowIdx * rowSkip + colIdx]);
        return sketch.estimate();
    }
};

// ----------------------------------------------------------------------------
// double <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct NumDistinctApprox<CSRMatrix<VT>> {
    static double apply(const CSRMatrix<VT> *arg, size_t K, int64_t seed, DCTX(ctx)) {
        KMVSketch<VT> sketch(K, seed);
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numRows == 0 || numCols == 0)
            return sketch.estimate();
        const size_t numNonZeros = arg->getNumNonZeros();
        if (numNonZeros) {
            // Start at the first row of this matrix, including for CSR views.
            const VT *values = arg->getValues(0);
            for (size_t valueIdx = 0; valueIdx < numNonZeros; ++valueIdx)
                sketch.insert(values[valueIdx]);
        }
        // Detect implicit zeros without overflowing the logical element count.
        if (numNonZeros / numCols < numRows)
            sketch.insert(VT(0));
        return sketch.estimate();
    }
};

// ----------------------------------------------------------------------------
// double <- Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct NumDistinctApprox<Matrix<VT>> {
    static double apply(const Matrix<VT> *arg, size_t K, int64_t seed, DCTX(ctx)) {
        KMVSketch<VT> sketch(K, seed);
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
            for (size_t colIdx = 0; colIdx < numCols; ++colIdx)
                sketch.insert(arg->get(rowIdx, colIdx));
        return sketch.estimate();
    }
};
