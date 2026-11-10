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
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DenseMatrix.h>

// Primary template declaration
template <class DT> struct TransferProperties {
    static void apply(const DT *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                      double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID,
                      ssize_t sparsityPatternID, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT>
void transferProperties(const DT *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                        bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                        double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID,
                        ssize_t sparsityPatternID, DCTX(ctx)) {
    TransferProperties<DT>::apply(arg, is_sparsity, sparsity, is_symmetric, symmetric, is_sortness, sortness,
                                  is_minValue, minValue, is_maxValue, maxValue, is_distinct, distinct,
                                  is_sparsityPatternID, sparsityPatternID, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template <typename VT> struct TransferProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, bool is_sparsity, double sparsity, bool is_symmetric,
                      int64_t symmetric, bool is_sortness, int64_t sortness, bool is_minValue, double minValue,
                      bool is_maxValue, double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID,
                      ssize_t sparsityPatternID, DCTX(ctx)) {
        auto mat = const_cast<DenseMatrix<VT> *>(arg);
        if (is_sparsity) {
            mat->is_sparsity = true;
            mat->sparsity = sparsity;
        }

        if (is_symmetric) {
            mat->is_symmetric = true;
            mat->symmetric = static_cast<BoolOrUnknown>(symmetric);
        }

        if (is_sortness) {
            mat->is_sortness = true;
            mat->sortness = static_cast<MatrixSortness>(sortness);
        }

        if (is_minValue) {
            mat->is_minValue = true;
            mat->minValue = minValue;
        }

        if (is_maxValue) {
            mat->is_maxValue = true;
            mat->maxValue = maxValue;
        }

        if (is_distinct) {
            mat->is_distinct = true;
            mat->distinct = distinct;
        }

        if (is_sparsityPatternID) {
            mat->is_sparsityPatternID = true;
            mat->sparsityPatternID = sparsityPatternID;
        }
    }
};

template <typename VT> struct TransferProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                      double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID,
                      ssize_t sparsityPatternID, DCTX(ctx)) {
        auto mat = const_cast<CSRMatrix<VT> *>(arg);
        if (is_sparsity) {
            mat->is_sparsity = true;
            mat->sparsity = sparsity;
        }

        if (is_symmetric) {
            mat->is_symmetric = true;
            mat->symmetric = static_cast<BoolOrUnknown>(symmetric);
        }

        if (is_sortness) {
            mat->is_sortness = true;
            mat->sortness = static_cast<MatrixSortness>(sortness);
        }

        if (is_minValue) {
            mat->is_minValue = true;
            mat->minValue = minValue;
        }

        if (is_maxValue) {
            mat->is_maxValue = true;
            mat->maxValue = maxValue;
        }

        if (is_distinct) {
            mat->is_distinct = true;
            mat->distinct = distinct;
        }

        if (is_sparsityPatternID) {
            mat->is_sparsityPatternID = true;
            mat->sparsityPatternID = sparsityPatternID;
        }
    }
};

template <typename VT> struct TransferProperties<Column<VT>> {
    static void apply(const Column<VT> *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                      double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID,
                      ssize_t sparsityPatternID, DCTX(ctx)) {
        auto col = const_cast<Column<VT> *>(arg);
        if (is_minValue) {
            col->is_minValue = true;
            col->minValue = minValue;
        }

        if (is_maxValue) {
            col->is_maxValue = true;
            col->maxValue = maxValue;
        }
    }
};
