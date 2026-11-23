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

#include <util/DistinctSketch.h>

#include <catch.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <set>

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {

    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numElements = 10000;
    std::srand(123456789);
    size_t expectedNumDistinct = 0;
    size_t approxResult = 0;

    SECTION("distinct") {
        std::vector<VT> v(numElements, 0);
        std::generate_n(v.begin(), numElements / 100, std::rand);

        auto mat10000 = genGivenVals<DT>(100, v);
        approxResult = numDistinctApprox(mat10000, 64, 1234567890, nullptr);
        expectedNumDistinct = 100;
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
    }
    SECTION("#distinct elements < K") {
        std::vector<VT> v(numElements, 0);
        v[0] = VT(1);
        auto twoDistinctValsMat = genGivenVals<DT>(100, v);

        approxResult = numDistinctApprox(twoDistinctValsMat, 64, 1234567890, nullptr);
        expectedNumDistinct = 2;
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
    size_t approxResult = 0;

    std::vector<VT> v(numElements, 0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(numRows, v);

    SECTION("full matrix") {
        auto fullSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows, 0, numCols);

        approxResult = numDistinctApprox(fullSubMat, 64, 1234567890, nullptr);
        expectedNumDistinct = numElements;
    }
    SECTION("view") {
        auto subMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100, 0, numCols);

        approxResult = numDistinctApprox(subMat, 64, 1234567890, nullptr);
        expectedNumDistinct = numElements / 100;
    }
    SECTION("#distinct elements < K") {
        auto smallSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100, 0, numCols / 10);

        approxResult = numDistinctApprox(smallSubMat, 64, 1234567890, nullptr);
        expectedNumDistinct = numElements / 1000;
    }

    // Allow +/-10% error. If the error is greater, the kernel is either wrongly parameterized (K to small) or the
    // algorithm broke.
    CHECK(Approx(approxResult).epsilon(1e-1) == expectedNumDistinct);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - view into CSRMatrix", TAG_KERNELS, (CSRMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numRows = 100;
    const size_t numCols = 100;
    const size_t numElements = numRows * numCols;
    size_t expectedNumDistinct = 0;
    size_t approxResult = 0;

    std::vector<VT> v(numElements, 0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(numRows, v);

    SECTION("full matrix") {
        auto fullSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows);
        approxResult = numDistinctApprox(fullSubMat, 64, 1234567890, nullptr);
        expectedNumDistinct = numElements;
    }
    SECTION("view") {
        auto subMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100);
        approxResult = numDistinctApprox(subMat, 64, 1234567890, nullptr);
        expectedNumDistinct = numElements / 100;
    }
    SECTION("#distinct elements < K") {
        auto smallSubMat = DataObjectFactory::create<DT>(mat10000, 0, numRows / 100);

        approxResult = numDistinctApprox(smallSubMat, 128, 1234567890, nullptr);
        expectedNumDistinct = numElements / 100;
    }

    // Allow +/-10% error. If the error is greater, the kernel is either wrongly parameterized (K to small) or the
    // algorithm broke.
    CHECK(Approx(approxResult).epsilon(1e-1) == expectedNumDistinct);
}

TEMPLATE_TEST_CASE("distinctSketch streaming estimate", TAG_KERNELS, int64_t, uint64_t, float, double) {
    using VT = TestType;
    using Sketch = property_acquisition::DistinctSketch<VT>;
    constexpr uint32_t seed = 1234567890;
    SECTION("invalid capacity") {
        CHECK_THROWS_AS(Sketch(0, seed), std::runtime_error);
        CHECK_THROWS_AS(Sketch(1, seed), std::runtime_error);
    }
    SECTION("empty and duplicate values") {
        Sketch sketch(64, seed);
        CHECK(sketch.estimate(0) == 0);
        sketch.insert(VT(0));
        sketch.insert(-VT(0));
        sketch.insert(VT(2));
        sketch.insert(VT(2));
        CHECK(sketch.estimate(4) == 2);
    }
    SECTION("saturated sketch agrees with a sorted-hash reference") {
        constexpr size_t size = 1000, k = 64;
        Sketch forward(k, seed), reverse(k, seed), repeated(k, seed);
        std::set<uint32_t> hashes;
        for (size_t i = 0; i < size; ++i) {
            VT value = VT(i);
            uint32_t hash;
            MurmurHash3_x86_32(&value, sizeof(VT), seed, &hash);
            hashes.insert(hash);
            forward.insert(value);
            reverse.insert(VT(size - i - 1));
            repeated.insert(value);
        }
        REQUIRE(hashes.size() >= k);
        const uint32_t kth = *std::next(hashes.begin(), k - 1);
        const double expected = std::min(double(size), double(k - 1) * double(UINT32_MAX) / double(kth));
        CHECK(forward.estimate(size) == Approx(expected));
        CHECK(reverse.estimate(size) == forward.estimate(size));
        CHECK(repeated.estimate(size) == forward.estimate(size));
        CHECK(std::isfinite(expected));
        CHECK(expected > k);
        CHECK(expected <= size);
    }
}
