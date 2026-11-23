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

#include <util/MurmurHash3.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace property_acquisition {
// Streaming K-minimum-values sketch for numeric values. Callers must exclude NaNs.
// Hash collisions make every result an estimate, even below capacity.
// estimate(size) requires the logical number of input elements, including repetitions.
template <typename VT> class DistinctSketch {
    size_t k;
    uint32_t seed;
    std::unordered_set<uint32_t> hashes;
    std::priority_queue<uint32_t> largest;

  public:
    DistinctSketch(size_t k, int64_t seed)
        : k(k), seed(static_cast<uint32_t>(
                    seed == -1 ? std::chrono::high_resolution_clock::now().time_since_epoch().count() : seed)) {
        if (k < 2)
            throw std::runtime_error("approximate distinct count requires K >= 2");
    }
    void insert(VT value) {
        if (value == VT(0))
            value = VT(0); // +0 and -0 are the same distinct value.
        uint32_t hash;
        MurmurHash3_x86_32(&value, sizeof(VT), seed, &hash);
        if (hashes.count(hash))
            return;
        if (hashes.size() < k) {
            hashes.insert(hash);
            largest.push(hash);
        } else if (hash < largest.top()) {
            hashes.erase(largest.top());
            largest.pop();
            hashes.insert(hash);
            largest.push(hash);
        }
    }
    double estimate(size_t size) const {
        if (hashes.size() < k)
            return static_cast<double>(hashes.size());
        if (largest.top() == 0)
            return static_cast<double>(size);
        return std::min(static_cast<double>(size),
                        double(k - 1) * double(std::numeric_limits<uint32_t>::max()) / double(largest.top()));
    }
};
} // namespace property_acquisition
