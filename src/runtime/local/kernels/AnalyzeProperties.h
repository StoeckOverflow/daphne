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

#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/IsSorted.h>
#include <runtime/local/kernels/IsSymmetric.h>
#include <runtime/local/kernels/NumDistinctApprox.h>
#include <runtime/local/kernels/NumDistinctCount.h>
#include <runtime/local/kernels/experimental/MaxAllSimd.h>
#include <runtime/local/kernels/experimental/MinAllSimd.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

// Internal request; independent from the flags identifying supplied transfer values.
struct AnalysisRequest {
    bool sparsity = false;
    bool symmetric = false;
    bool sortedness = false;
    bool minValue = false;
    bool maxValue = false;
    bool distinct = false;
    bool sparsityPattern = false;
};

template <class DT> struct AnalyzeProperties {
    static void apply(const DT *arg, const AnalysisRequest &request, DCTX(ctx)) = delete;
};

template <typename VT> struct AnalyzeProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const AnalysisRequest &request, DCTX(ctx)) {
        auto mat = const_cast<DenseMatrix<VT> *>(arg);
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const auto &userConfig = ctx->getUserConfig();
        bool needSparsity = request.sparsity && !mat->is_sparsity;
        bool needSymmetric = request.symmetric && !mat->is_symmetric;
        bool needSortness = request.sortedness && !mat->is_sortness;
        bool needMin = request.minValue && !mat->is_minValue;
        bool needMax = request.maxValue && !mat->is_maxValue;
        bool needDistinct = request.distinct && !mat->is_distinct;

        bool analyzedAny = false;
        auto startTime = std::chrono::steady_clock::now();

        // Step 4: Perform runtime property analysis

        // Analyze sparsity
        if (needSparsity) {
            const VT *values = arg->getValues();
            size_t totalElements = numRows * numCols;
            size_t numZeros = 0;

            for (size_t i = 0; i < totalElements; ++i) {
                if (values[i] == static_cast<VT>(0)) {
                    numZeros++;
                }
            }

            mat->sparsity = static_cast<double>(totalElements - numZeros) / static_cast<double>(totalElements);
            mat->is_sparsity = true;
            analyzedAny = true;
        }

        if constexpr (std::is_arithmetic_v<VT>) {
            if (needSymmetric) {
                mat->symmetric = numRows == numCols && isSymmetric(arg, ctx) ? BoolOrUnknown::True
                                                                         : BoolOrUnknown::False;
                mat->is_symmetric = true;
                analyzedAny = true;
            }
        }

        if (needSortness) {
            mat->sortness = static_cast<MatrixSortness>(isSorted(arg, ctx));
            mat->is_sortness = true;
            analyzedAny = true;
        }

        if constexpr (std::is_arithmetic_v<VT> && !std::is_same_v<VT, bool>) {
            if (needMin || needMax) {
                bool useSIMD = userConfig.adaptiveAnalyze &&
                               userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd;

                // Try SIMD version first if enabled
                if (useSIMD) {
                    try {
                        if (needMin) {
                            mat->minValue = minAllSimd<DenseMatrix<VT>, VT>(arg, ctx);
                            mat->is_minValue = true;
                            analyzedAny = true;
                        }
                        if (needMax) {
                            mat->maxValue = maxAllSimd<DenseMatrix<VT>, VT>(arg, ctx);
                            mat->is_maxValue = true;
                            analyzedAny = true;
                        }
                    } catch (const std::runtime_error &) {
                        // SIMD failed (e.g., wrong size), fall back to regular version
                        if (needMin) {
                            mat->minValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MIN, arg, ctx);
                            mat->is_minValue = true;
                            analyzedAny = true;
                        }
                        if (needMax) {
                            mat->maxValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MAX, arg, ctx);
                            mat->is_maxValue = true;
                            analyzedAny = true;
                        }
                    }
                } else {
                    // Use regular version
                    if (needMin) {
                        mat->minValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MIN, arg, ctx);
                        mat->is_minValue = true;
                        analyzedAny = true;
                    }
                    if (needMax) {
                        mat->maxValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MAX, arg, ctx);
                        mat->is_maxValue = true;
                        analyzedAny = true;
                    }
                }
            }
        }

        if (needDistinct) {
            bool computed = false;
            auto analyzeMode = userConfig.adaptiveAnalyzeMode;

            // Option A: Approximate counting
            if (userConfig.adaptiveAnalyze && analyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Approx) {
                size_t K = userConfig.approxNDistinctK;
                int64_t seed = userConfig.approxSeed;
                size_t approxCount = numDistinctApprox<DenseMatrix<VT>>(arg, K, seed, ctx);
                mat->distinct = static_cast<ssize_t>(approxCount);
                mat->is_distinct = true;
                computed = true;
            }

            // Option B: Early-abort counting
            if (!computed && userConfig.adaptiveAnalyze &&
                analyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::EarlyAbort) {
                // Calculate threshold
                size_t threshold = 0;
                if (userConfig.mapDistinctThresholdIsRelative) {
                    double fraction = userConfig.mapDistinctThresholdFraction;
                    threshold = static_cast<size_t>(fraction * static_cast<double>(numRows * numCols));
                } else {
                    threshold = userConfig.mapDistinctThresholdAbsolute;
                }
                if (threshold == 0)
                    threshold = 1;

                // Count distinct values with early abort
                size_t totalElements = numRows * numCols;
                std::unordered_set<VT> uniqueValues;
                uniqueValues.reserve(std::min(threshold, totalElements));

                const VT *values = arg->getValues();
                bool exceededThreshold = false;
                for (size_t i = 0; i < totalElements; ++i) {
                    uniqueValues.insert(values[i]);
                    if (uniqueValues.size() > threshold) {
                        exceededThreshold = true;
                        break;
                    }
                }

                if (exceededThreshold) {
                    mat->distinct = static_cast<ssize_t>(threshold + 1);
                } else {
                    mat->distinct = static_cast<ssize_t>(uniqueValues.size());
                }
                mat->is_distinct = true;
                computed = true;
            }

            // Option C: Exact counting
            if (!computed) {
                mat->distinct = numDistinctCount<DenseMatrix<VT>>(arg, ctx);
                mat->is_distinct = true;
            }
            analyzedAny = true;
        }

        // Note: sparsity pattern analysis not applicable for DenseMatrix

        // Exit early if nothing was analyzed
        if (!analyzedAny)
            return;

        // Record analysis time and statistics
        auto endTime = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();

        // Retain the legacy benchmark log label for existing experiment parsers.
        if (ctx && ctx->getUserConfig().enable_statistics)
            std::cerr << "[KERNEL_TIME] TransferProperties: " << std::fixed << std::setprecision(6) << seconds
                  << " seconds (DenseMatrix " << numRows << "x" << numCols << ")" << std::endl;
    }
};

// CSR discovery was not implemented in the original runtime. Preserve that
// behavior: unsupported requests do not manufacture known metadata.
template <typename VT> struct AnalyzeProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *, const AnalysisRequest &, DCTX(ctx)) {}
};

// Columns currently support acquisition of bounds only.
template <typename VT> struct AnalyzeProperties<Column<VT>> {
    static void apply(const Column<VT> *arg, const AnalysisRequest &request, DCTX(ctx)) {
        auto col = const_cast<Column<VT> *>(arg);
        const size_t numRows = arg->getNumRows();
        const auto &userConfig = ctx->getUserConfig();
        bool needMin = request.minValue && !col->is_minValue;
        bool needMax = request.maxValue && !col->is_maxValue;

        if (!needMin && !needMax) {
            return; // nothing to analyze
        }

        bool analyzedAny = false;
        auto startTime = std::chrono::steady_clock::now();

        // Analyze min/max values for Column
        if constexpr (std::is_arithmetic_v<VT> && !std::is_same_v<VT, bool>) {
            if (needMin || needMax) {
                bool useSIMD = userConfig.adaptiveAnalyze &&
                               userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd;

                // SIMD implementation for int64_t type with AVX2
                if (useSIMD && std::is_same_v<VT, int64_t>) {
                    if (numRows > 0) {
                        // Lambda with target attribute for AVX2
                        auto simdMinMax = [](const int64_t *values, size_t numRows) __attribute__((target("avx2")))
                                              ->std::pair<int64_t, int64_t> {

                            const size_t simdWidth = 4;
                            const size_t simdIterations = numRows / simdWidth;

                            // Initialize with first value
                            __m256i minVec = _mm256_set1_epi64x(values[0]);
                            __m256i maxVec = _mm256_set1_epi64x(values[0]);

                            // SIMD main loop - use comparison and blend for int64 min/max
                            for (size_t i = 0; i < simdIterations; i++) {
                                __m256i dataVec = _mm256_loadu_si256((__m256i *)&values[i * simdWidth]);

                                // Min: if data < min, use data, else use min
                                __m256i cmpMin = _mm256_cmpgt_epi64(minVec, dataVec);
                                minVec = _mm256_blendv_epi8(minVec, dataVec, cmpMin);

                                // Max: if data > max, use data, else use max
                                __m256i cmpMax = _mm256_cmpgt_epi64(dataVec, maxVec);
                                maxVec = _mm256_blendv_epi8(maxVec, dataVec, cmpMax);
                            }

                            // Horizontal reduction
                            int64_t minArr[4], maxArr[4];
                            _mm256_storeu_si256((__m256i *)minArr, minVec);
                            _mm256_storeu_si256((__m256i *)maxArr, maxVec);

                            int64_t minVal = std::min({minArr[0], minArr[1], minArr[2], minArr[3]});
                            int64_t maxVal = std::max({maxArr[0], maxArr[1], maxArr[2], maxArr[3]});

                            // Process tail elements
                            size_t tailStart = simdIterations * simdWidth;
                            for (size_t i = tailStart; i < numRows; i++) {
                                if (values[i] < minVal)
                                    minVal = values[i];
                                if (values[i] > maxVal)
                                    maxVal = values[i];
                            }

                            return {minVal, maxVal};
                        };

                        const int64_t *values = reinterpret_cast<const int64_t *>(arg->getValues());
                        auto [minVal, maxVal] = simdMinMax(values, numRows);

                        if (needMin) {
                            col->minValue = static_cast<double>(minVal);
                            col->is_minValue = true;
                            analyzedAny = true;
                        }
                        if (needMax) {
                            col->maxValue = static_cast<double>(maxVal);
                            col->is_maxValue = true;
                            analyzedAny = true;
                        }
                    }
                }
                // Scalar fallback for non-int64_t types or when SIMD disabled
                else {
                    const VT *values = arg->getValues();
                    if (numRows > 0) {
                        VT minVal = values[0];
                        VT maxVal = values[0];

                        for (size_t i = 1; i < numRows; i++) {
                            if (values[i] < minVal)
                                minVal = values[i];
                            if (values[i] > maxVal)
                                maxVal = values[i];
                        }

                        if (needMin) {
                            col->minValue = static_cast<double>(minVal);
                            col->is_minValue = true;
                            analyzedAny = true;
                        }
                        if (needMax) {
                            col->maxValue = static_cast<double>(maxVal);
                            col->is_maxValue = true;
                            analyzedAny = true;
                        }
                    }
                }
            }
        }

        if (!analyzedAny)
            return;

        // Record analysis time and statistics
        auto endTime = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();

        // Retain the legacy benchmark log label for existing experiment parsers.
        if (ctx && ctx->getUserConfig().enable_statistics)
            std::cerr << "[KERNEL_TIME] TransferProperties: " << std::fixed << std::setprecision(6) << seconds
                  << " seconds (Column " << numRows << ")" << std::endl;
    }
};
