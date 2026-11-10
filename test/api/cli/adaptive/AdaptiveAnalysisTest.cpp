/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <api/cli/Utils.h>
#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>

namespace {
const std::string dirPath = "test/api/cli/adaptive/";

template <typename... Args>
void checkAdaptive(const char *script, const std::string &expected, Args... args) {
    const auto path = dirPath + script;
    INFO(script);
    compareDaphneToStr(expected, path, args...);
    compareDaphneToStr(expected, path, "--adaptiveAnalyze=exact", args...);
}
} // namespace

TEST_CASE("Adaptive MatMul dense", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_dense.daphne", "DenseMatrix(2x2, double)\n19 22\n43 50\n");
}

TEST_CASE("Adaptive MatMul sparse", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_sparse.daphne", "DenseMatrix(2x2, double)\n2 4\n3 6\n");
}

TEST_CASE("Adaptive MatMul transposed", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_transposed.daphne", "DenseMatrix(2x2, double)\n2 2\n3 3\n");
}

TEST_CASE("Adaptive k-means empty cluster", TAG_ADAPTIVE TAG_ALGORITHMS) {
    checkAdaptive("kmeans_empty_cluster.daphne", "DenseMatrix(3x2, double)\n0 1\n10 11\n100 100\n");
}

TEST_CASE("Adaptive MatMul known density", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_known_density.daphne", "DenseMatrix(2x2, double)\n100 100\n100 100\n");
}

TEST_CASE("Adaptive MatMul across branches", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_branch.daphne", "DenseMatrix(2x2, double)\n2 2\n3 3\nDenseMatrix(2x2, double)\n2 2\n3 3\n");
}

TEST_CASE("Adaptive analysis FileCheck", TAG_ADAPTIVE TAG_MATMUL) {
    std::stringstream out, err;
    const int status = runLIT(out, err, "test/util/", "--filter=adaptive_");
    INFO(out.str());
    INFO(err.str());
    CHECK(status == StatusCode::SUCCESS);
}

TEST_CASE("Adaptive comparison", TAG_ADAPTIVE TAG_OPERATIONS) {
    checkAdaptive("comparison.daphne", "1\n");
    checkAdaptive("comparison.daphne", "1\n", "--vec-restricted");
}

TEST_CASE("Adaptive map", TAG_ADAPTIVE TAG_SECONDORDER) {
    checkAdaptive("map.daphne", "10\n");
    checkAdaptive("map.daphne", "10\n", "--map-distinct-threshold-adaptive");
}

TEST_CASE("Adaptive columnar join", TAG_ADAPTIVE TAG_SQL) {
    checkAdaptive("join.daphne", "Frame(2x2, [a.value:int64_t, b.value:int64_t])\n20 40\n30 50\n", "--columnar");
}

TEST_CASE("Automatic property analysis", TAG_ADAPTIVE TAG_OPERATIONS) {
    checkAdaptive("comparison.daphne", "1\n", "--automaticallyAnalyzeEverything");
    checkAdaptive("matmul_branch.daphne", "DenseMatrix(2x2, double)\n2 2\n3 3\nDenseMatrix(2x2, double)\n2 2\n3 3\n",
                  "--automaticallyAnalyzeEverything");
}

TEST_CASE("Explicit analysis with runtime conditions", TAG_ADAPTIVE TAG_OPERATIONS) {
    compareDaphneToStr("DenseMatrix(2x2, double)\n2 2\n3 3\nDenseMatrix(2x2, double)\n4 4\n6 6\n",
                      dirPath + "analyze_dynamic.daphne");
}

TEST_CASE("Adaptive path diagnostics are opt-in", TAG_ADAPTIVE TAG_OPERATIONS) {
    std::stringstream out, err;
    const auto path = dirPath + "analyze_dynamic.daphne";
    const int status = runDaphne(out, err, "--statistics", path.c_str());
    INFO(err.str());
    REQUIRE(status == StatusCode::SUCCESS);
    const auto dense = err.str().find("seconds (dense-blas,");
    const auto sparse = err.str().find("seconds (sparse-aware,");
    REQUIRE(dense != std::string::npos);
    REQUIRE(sparse != std::string::npos);
    CHECK(dense < sparse);
    CHECK_THAT(err.str(), Catch::Contains("[KERNEL_TIME] TransferProperties:"));
}
