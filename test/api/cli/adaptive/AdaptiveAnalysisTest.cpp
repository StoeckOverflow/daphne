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

void checkAdaptive(const char *script, const std::string &expected, const char *adaptivePath = nullptr) {
    for (bool adaptive : {false, true}) {
        DYNAMIC_SECTION(script << (adaptive ? " exact analysis" : " default")) {
            std::stringstream out, err;
            const auto path = dirPath + script;
            const int status = adaptive ? runDaphne(out, err, "--adaptiveAnalyze=exact", path.c_str())
                                        : runDaphne(out, err, path.c_str());
            INFO(err.str());
            REQUIRE(status == StatusCode::SUCCESS);
            CHECK(out.str() == expected);
            if (adaptivePath) {
                const std::string marker = adaptive ? adaptivePath : "dense-blas";
                CHECK_THAT(err.str(), Catch::Contains("seconds (" + marker + ","));
                const std::string other = marker == "dense-blas" ? "sparse-aware" : "dense-blas";
                CHECK_THAT(err.str(), !Catch::Contains("seconds (" + other + ","));
            }
        }
    }
}
} // namespace

TEST_CASE("Adaptive MatMul dense", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_dense.daphne", "DenseMatrix(2x2, double)\n19 22\n43 50\n", "dense-blas");
}

TEST_CASE("Adaptive MatMul sparse", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_sparse.daphne", "DenseMatrix(2x2, double)\n2 4\n3 6\n", "sparse-aware");
}

TEST_CASE("Adaptive MatMul transposed", TAG_ADAPTIVE TAG_MATMUL) {
    checkAdaptive("matmul_transposed.daphne", "DenseMatrix(2x2, double)\n2 2\n3 3\n", "dense-blas");
}

TEST_CASE("Adaptive k-means empty cluster", TAG_ADAPTIVE TAG_ALGORITHMS) {
    checkAdaptive("kmeans_empty_cluster.daphne", "DenseMatrix(3x2, double)\n0 1\n10 11\n100 100\n");
}
