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

#ifndef SRC_IR_DAPHNEIR_DAPHNETYPESTORAGE_H
#define SRC_IR_DAPHNEIR_DAPHNETYPESTORAGE_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>

namespace mlir::daphne {

// Forward declarations of enums
enum class MatrixRepresentation;
enum class BoolOrUnknown;
enum class MatrixSortness;
namespace detail {

struct MatrixTypeStorage : public ::mlir::TypeStorage {
    // TODO: adapt epsilon for equality check (I think the only use is saving
    // memory for the MLIR-IR representation of this type)
    //  the choosen epsilon directly defines how accurate our sparsity inference
    //  can be
    constexpr static const double epsilon = 1e-6;
    MatrixTypeStorage(::mlir::Type elementType, ssize_t numRows, ssize_t numCols, double sparsity,
                      MatrixRepresentation representation, BoolOrUnknown symmetric, MatrixSortness sortness,
                      std::optional<double> minValue, std::optional<double> maxValue, ssize_t distinct,
                      ssize_t sparsityPatternID)
        : elementType(elementType), numRows(numRows), numCols(numCols), sparsity(sparsity),
          representation(representation), symmetric(symmetric), sortness(sortness), minValue(minValue),
          maxValue(maxValue), distinct(distinct), sparsityPatternID(sparsityPatternID) {}

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, ssize_t, ssize_t, double, MatrixRepresentation, BoolOrUnknown,
                             MatrixSortness, std::optional<double>, std::optional<double>, ssize_t, ssize_t>;
    bool operator==(const KeyTy &tblgenKey) const {
        if (!(elementType == std::get<0>(tblgenKey)))
            return false;
        if (numRows != std::get<1>(tblgenKey))
            return false;
        if (numCols != std::get<2>(tblgenKey))
            return false;
        if (std::fabs(sparsity - std::get<3>(tblgenKey)) >= epsilon)
            return false;
        if (representation != std::get<4>(tblgenKey))
            return false;
        if (symmetric != std::get<5>(tblgenKey))
            return false;
        if (sortness != std::get<6>(tblgenKey))
            return false;
        // Compare optionals with epsilon tolerance when both present
        const auto &minKey = std::get<7>(tblgenKey);
        const auto &maxKey = std::get<8>(tblgenKey);
        if (minValue.has_value() != minKey.has_value())
            return false;
        if (maxValue.has_value() != maxKey.has_value())
            return false;
        if (minValue && minKey && std::fabs(*minValue - *minKey) >= epsilon)
            return false;
        if (maxValue && maxKey && std::fabs(*maxValue - *maxKey) >= epsilon)
            return false;
        if (distinct != std::get<9>(tblgenKey))
            return false;
        if (sparsityPatternID != std::get<10>(tblgenKey))
            return false;
        return true;
    }
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
        auto float_hashable = static_cast<ssize_t>(std::get<3>(tblgenKey) / epsilon);
        // Quantize optionals to integer buckets; use sentinels for nullopt
        const auto &minKey = std::get<7>(tblgenKey);
        const auto &maxKey = std::get<8>(tblgenKey);
        const ssize_t min_hashable = minKey.has_value() ? static_cast<ssize_t>(std::llround(*minKey / epsilon))
                                                        : std::numeric_limits<ssize_t>::min();
        const ssize_t max_hashable = maxKey.has_value() ? static_cast<ssize_t>(std::llround(*maxKey / epsilon))
                                                        : std::numeric_limits<ssize_t>::min() + 1;
        return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey),
                                    float_hashable, std::get<4>(tblgenKey), std::get<5>(tblgenKey),
                                    std::get<6>(tblgenKey), min_hashable, max_hashable, std::get<9>(tblgenKey),
                                    std::get<10>(tblgenKey));
    }

    /// Define a construction method for creating a new instance of this
    /// storage.
    static MatrixTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
        auto elementType = std::get<0>(tblgenKey);
        auto numRows = std::get<1>(tblgenKey);
        auto numCols = std::get<2>(tblgenKey);
        auto sparsity = std::get<3>(tblgenKey);
        auto representation = std::get<4>(tblgenKey);
        auto symmetric = std::get<5>(tblgenKey);
        auto sortness = std::get<6>(tblgenKey);
        auto minValue = std::get<7>(tblgenKey);
        auto maxValue = std::get<8>(tblgenKey);
        auto distinct = std::get<9>(tblgenKey);
        auto sparsityPatternID = std::get<10>(tblgenKey);

        return new (allocator.allocate<MatrixTypeStorage>())
            MatrixTypeStorage(elementType, numRows, numCols, sparsity, representation, symmetric, sortness, minValue,
                              maxValue, distinct, sparsityPatternID);
    }
    ::mlir::Type elementType;
    ssize_t numRows;
    ssize_t numCols;
    double sparsity;
    MatrixRepresentation representation;
    BoolOrUnknown symmetric;
    MatrixSortness sortness;
    std::optional<double> minValue;
    std::optional<double> maxValue;
    ssize_t distinct;
    ssize_t sparsityPatternID;
};

struct ColumnTypeStorage : public ::mlir::TypeStorage {
    constexpr static const double epsilon = 1e-6;

    ColumnTypeStorage(::mlir::Type valueType, ssize_t numRows, std::optional<double> minValue,
                      std::optional<double> maxValue)
        : valueType(valueType), numRows(numRows), minValue(minValue), maxValue(maxValue) {}

    using KeyTy = std::tuple<::mlir::Type, ssize_t, std::optional<double>, std::optional<double>>;

    bool operator==(const KeyTy &tblgenKey) const {
        if (!(valueType == std::get<0>(tblgenKey)))
            return false;
        if (numRows != std::get<1>(tblgenKey))
            return false;
        // Compare optionals with epsilon tolerance when both present
        const auto &minKey = std::get<2>(tblgenKey);
        const auto &maxKey = std::get<3>(tblgenKey);
        if (minValue.has_value() != minKey.has_value())
            return false;
        if (maxValue.has_value() != maxKey.has_value())
            return false;
        if (minValue && minKey && std::fabs(*minValue - *minKey) >= epsilon)
            return false;
        if (maxValue && maxKey && std::fabs(*maxValue - *maxKey) >= epsilon)
            return false;
        return true;
    }

    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
        const auto &minKey = std::get<2>(tblgenKey);
        const auto &maxKey = std::get<3>(tblgenKey);
        const ssize_t min_hashable = minKey.has_value() ? static_cast<ssize_t>(std::llround(*minKey / epsilon))
                                                        : std::numeric_limits<ssize_t>::min();
        const ssize_t max_hashable = maxKey.has_value() ? static_cast<ssize_t>(std::llround(*maxKey / epsilon))
                                                        : std::numeric_limits<ssize_t>::min() + 1;
        return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), min_hashable, max_hashable);
    }

    static ColumnTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
        auto valueType = std::get<0>(tblgenKey);
        auto numRows = std::get<1>(tblgenKey);
        auto minValue = std::get<2>(tblgenKey);
        auto maxValue = std::get<3>(tblgenKey);
        return new (allocator.allocate<ColumnTypeStorage>()) ColumnTypeStorage(valueType, numRows, minValue, maxValue);
    }

    ::mlir::Type valueType;
    ssize_t numRows;
    std::optional<double> minValue;
    std::optional<double> maxValue;
};

} // namespace detail
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNETYPESTORAGE_H
