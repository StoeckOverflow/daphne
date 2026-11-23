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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/NumDistinctApprox.h>
#include <runtime/local/kernels/RandMatrix.h>

#include <tags.h>

#include <catch.hpp>

#include <cstddef>
#include <cstdlib>
#include <limits>

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {

    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numElements = 10000;
    std::srand(123456789);
    size_t expectedNumDistinct = 0;
    double approxResult = 0;

    SECTION("distinct") {
        std::vector<VT> v(numElements, 0);
        std::generate_n(v.begin(), numElements / 100, std::rand);

        auto mat10000 = genGivenVals<DT>(100, v);
        approxResult = numDistinctApprox(mat10000, 64, 1234567890, nullptr);
        expectedNumDistinct = 100;
        DataObjectFactory::destroy(mat10000);
    }
    SECTION("distinct leading 100 zeros") {
        std::vector<VT> v(numElements, 0);
        std::srand(123456789);

        auto it = v.begin();
        std::advance(it, 100);
        std::generate_n(it, numElements / 100, std::rand);

        auto matZerosAtStart = genGivenVals<DT>(100, v);
        approxResult = numDistinctApprox(matZerosAtStart, 64, 1234567890, nullptr);
        expectedNumDistinct = 100;
        DataObjectFactory::destroy(matZerosAtStart);
    }
    SECTION("#distinct elements < K") {
        std::vector<VT> v(numElements, 0);
        v[0] = VT(1);
        auto twoDistinctValsMat = genGivenVals<DT>(100, v);

        approxResult = numDistinctApprox(twoDistinctValsMat, 64, 1234567890, nullptr);
        expectedNumDistinct = 2;
        DataObjectFactory::destroy(twoDistinctValsMat);
    }

    SECTION("empty matrix and invalid capacity") {
        auto empty = genGivenVals<DT>(1, {});
        approxResult = numDistinctApprox(empty, 64, 1234567890, nullptr);
        expectedNumDistinct = 0;
        CHECK_THROWS_AS(numDistinctApprox(empty, 0, 1234567890, nullptr), std::runtime_error);
        CHECK_THROWS_AS(numDistinctApprox(empty, 1, 1234567890, nullptr), std::runtime_error);
        DataObjectFactory::destroy(empty);
    }
    SECTION("duplicates after capacity is reached") {
        auto atCapacity = genGivenVals<DT>(1, {1, 2});
        auto distinct = genGivenVals<DT>(1, {1, 2, 3});
        auto repeated = genGivenVals<DT>(2, {3, 2, 1, 1, 2, 3});
        approxResult = numDistinctApprox(atCapacity, 2, 1234567890, nullptr);
        expectedNumDistinct = 2;
        CHECK(approxResult == expectedNumDistinct);
        CHECK(numDistinctApprox(repeated, 2, 1234567890, nullptr) ==
              numDistinctApprox(distinct, 2, 1234567890, nullptr));
        DataObjectFactory::destroy(atCapacity, distinct, repeated);
    }

    // Allow +/-10% error. If the error is greater, the kernel is either wrongly parameterized (K to small) or the
    // algorithm broke.
    CHECK(Approx(approxResult).epsilon(1e-1) == expectedNumDistinct);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - view into DenseMatrix", TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numRows = 100;
    const size_t numCols = 100;
    const size_t numElements = numRows * numCols;
    size_t expectedNumDistinct = 0;
    double approxResult = 0;

    std::vector<VT> v(numElements, 0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(numRows, v);

    SECTION("full matrix") {
        auto fullSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows, 0, numCols);

        approxResult = numDistinctApprox(fullSubMat, 1024, 1234567890, nullptr);
        expectedNumDistinct = numElements;
        DataObjectFactory::destroy(fullSubMat);
    }
    SECTION("view") {
        auto subMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100, 0, numCols);

        approxResult = numDistinctApprox(subMat, 128, 1234567890, nullptr);
        expectedNumDistinct = numElements / 100;
        DataObjectFactory::destroy(subMat);
    }
    SECTION("#distinct elements < K") {
        auto smallSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100, 0, numCols / 10);

        approxResult = numDistinctApprox(smallSubMat, 64, 1234567890, nullptr);
        expectedNumDistinct = numElements / 1000;
        DataObjectFactory::destroy(smallSubMat);
    }

    SECTION("view with row stride and nonzero offset") {
        auto matrix = genGivenVals<DT>(3, {99, 99, 99, 99, 99, 0, 2, 99, 99, 2, 0, 99});
        auto view = DataObjectFactory::create<DT>(matrix, 1, 3, 1, 3);
        approxResult = numDistinctApprox(view, 64, 1234567890, nullptr);
        expectedNumDistinct = 2;
        DataObjectFactory::destroy(view, matrix);
    }

    // Allow +/-10% error. If the error is greater, the kernel is either wrongly parameterized (K to small) or the
    // algorithm broke.
    CHECK(Approx(approxResult).epsilon(1e-1) == expectedNumDistinct);
    DataObjectFactory::destroy(mat10000);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - view into CSRMatrix", TAG_KERNELS, (CSRMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numRows = 100;
    const size_t numCols = 100;
    const size_t numElements = numRows * numCols;
    size_t expectedNumDistinct = 0;
    double approxResult = 0;

    std::vector<VT> v(numElements, 0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(numRows, v);

    SECTION("full matrix") {
        auto fullSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows);
        approxResult = numDistinctApprox(fullSubMat, 1024, 1234567890, nullptr);
        expectedNumDistinct = numElements;
        DataObjectFactory::destroy(fullSubMat);
    }
    SECTION("view") {
        auto subMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100);
        approxResult = numDistinctApprox(subMat, 128, 1234567890, nullptr);
        expectedNumDistinct = numElements / 100;
        DataObjectFactory::destroy(subMat);
    }
    SECTION("#distinct elements < K") {
        auto smallSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100);

        approxResult = numDistinctApprox(smallSubMat, 128, 1234567890, nullptr);
        expectedNumDistinct = numElements / 100;
        DataObjectFactory::destroy(smallSubMat);
    }

    SECTION("view with nonzero row offset") {
        auto matrix = genGivenVals<DT>(3, {99, 99, 99, 0, 2, 0, 2, 0, 2});
        auto view = DataObjectFactory::create<DT>(matrix, 1, 3);
        approxResult = numDistinctApprox(view, 64, 1234567890, nullptr);
        expectedNumDistinct = 2;
        DataObjectFactory::destroy(view, matrix);
    }

    SECTION("nonempty matrix without stored values") {
        auto matrix = genGivenVals<DT>(2, {0, 0, 0, 0});
        approxResult = numDistinctApprox(matrix, 64, 1234567890, nullptr);
        expectedNumDistinct = 1;
        DataObjectFactory::destroy(matrix);
    }

    SECTION("fully stored view with explicit zero") {
        auto matrix = genGivenVals<DT>(3, {99, 99, 1, 2, 1, 2});
        auto view = DataObjectFactory::create<DT>(matrix, 1, 3);
        view->getValues(0)[0] = VT(0);
        approxResult = numDistinctApprox(view, 64, 1234567890, nullptr);
        expectedNumDistinct = 3;
        DataObjectFactory::destroy(view, matrix);
    }

    // Allow +/-10% error. If the error is greater, the kernel is either wrongly parameterized (K to small) or the
    // algorithm broke.
    CHECK(Approx(approxResult).epsilon(1e-1) == expectedNumDistinct);
    DataObjectFactory::destroy(mat10000);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - floating-point values", TAG_KERNELS, (DATA_TYPES), (double, float)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto matrix =
        genGivenVals<DT>(1, {VT(0), -VT(0), std::numeric_limits<VT>::infinity(), -std::numeric_limits<VT>::infinity()});
    CHECK(numDistinctApprox(matrix, 64, 1234567890, nullptr) == 3);
    matrix->set(0, 2, std::numeric_limits<VT>::quiet_NaN());
    CHECK_THROWS_AS(numDistinctApprox(matrix, 64, 1234567890, nullptr), std::runtime_error);
    DataObjectFactory::destroy(matrix);
}
