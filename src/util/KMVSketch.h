/*
 * Copyright 2026 The DAPHNE Consortium
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

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

template <typename VT> class KMVSketch {
    size_t k;
    uint32_t seed;
    std::unordered_set<uint64_t> hashes;
    std::priority_queue<uint64_t> largest;
    bool estimating = false;

  public:
    KMVSketch(size_t k, int64_t seed)
        : k(k), seed(static_cast<uint32_t>(
                    seed == -1 ? std::chrono::high_resolution_clock::now().time_since_epoch().count() : seed)) {
        if (k < 2)
            throw std::runtime_error("approximate distinct count requires K >= 2");
    }

    void insert(VT value) {
        if constexpr (std::is_floating_point_v<VT>)
            if (std::isnan(value))
                throw std::runtime_error("KMV distinct count does not support NaN values");
        if (value == VT(0))
            value = VT(0); // +0 and -0 are the same distinct value.
        uint64_t output[2];
        MurmurHash3_x64_128(&value, sizeof(VT), seed, output);
        const uint64_t hash = output[0];
        if (hashes.size() == k && hash >= largest.top()) {
            // Equality is a retained duplicate. A larger hash exceeds capacity.
            if (hash > largest.top())
                estimating = true;
            return;
        }
        if (hashes.count(hash))
            return;
        if (hashes.size() == k) {
            estimating = true;
            hashes.erase(largest.top());
            largest.pop();
        }
        hashes.insert(hash);
        largest.push(hash);
    }

    double estimate() const {
        if (!estimating)
            return static_cast<double>(hashes.size());
        // k >= 2 distinct unsigned hashes guarantee a nonzero threshold.
        return double(k - 1) * double(std::numeric_limits<uint64_t>::max()) / double(largest.top());
    }
};
