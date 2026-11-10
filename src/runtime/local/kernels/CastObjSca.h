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
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <typename VTRes, class DTArg> struct CastObjSca {
    static VTRes apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename VTRes, class DTArg> VTRes castObjSca(const DTArg *arg, DCTX(ctx)) {
    return CastObjSca<VTRes, DTArg>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Scalar <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct CastObjSca<VTRes, DenseMatrix<VTArg>> {
    static VTRes apply(const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numCols != 1 || numRows != 1)
            throw std::runtime_error("Cast matrix to scalar: matrix shape should be 1x1");
        return static_cast<VTRes>(*arg->getValues());
    }
};

// ----------------------------------------------------------------------------
// Scalar <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct CastObjSca<VTRes, CSRMatrix<VTArg>> {
    static VTRes apply(const CSRMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numCols != 1 || numRows != 1)
            throw std::runtime_error("Cast matrix to scalar: matrix shape should be 1x1");
        return arg->getNumNonZeros(0) ? *arg->getValues(0) : 0;
    }
};

// ----------------------------------------------------------------------------
//  Scalar <- Frame
// ----------------------------------------------------------------------------

template <typename VTRes> struct CastObjSca<VTRes, Frame> {
    static VTRes apply(const Frame *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        if (numCols != 1 || numRows != 1)
            throw std::runtime_error("Cast frame to scalar: frame shape should be 1x1");

        const void *resVal = arg->getColumnRaw(0);
        switch (arg->getColumnType(0)) {
        case ValueTypeCode::SI8:
            return static_cast<VTRes>(*static_cast<const int8_t *>(resVal));
        case ValueTypeCode::SI32:
            return static_cast<VTRes>(*static_cast<const int32_t *>(resVal));
        case ValueTypeCode::SI64:
            return static_cast<VTRes>(*static_cast<const int64_t *>(resVal));
        case ValueTypeCode::UI8:
            return static_cast<VTRes>(*static_cast<const uint8_t *>(resVal));
        case ValueTypeCode::UI16:
            return static_cast<VTRes>(*static_cast<const uint16_t *>(resVal));
        case ValueTypeCode::UI32:
            return static_cast<VTRes>(*static_cast<const uint32_t *>(resVal));
        case ValueTypeCode::UI64:
            return static_cast<VTRes>(*static_cast<const uint64_t *>(resVal));
        case ValueTypeCode::F32:
            return static_cast<VTRes>(*static_cast<const float *>(resVal));
        case ValueTypeCode::F64:
            return static_cast<VTRes>(*static_cast<const double *>(resVal));
        case ValueTypeCode::I1:
            return static_cast<VTRes>(*static_cast<const bool *>(resVal));
        default:
            throw std::runtime_error("CastObjSca::apply: unsupported value type code");
        }
    }
};

// ----------------------------------------------------------------------------
// Scalar <- Column
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct CastObjSca<VTRes, Column<VTArg>> {
    static VTRes apply(const Column<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        if (numRows != 1)
            throw std::runtime_error("cast column to scalar: column must have exactly one element");
        return static_cast<VTRes>(*arg->getValues());
    }
};