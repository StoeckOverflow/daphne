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

#include <cstdint>
#include <optional>
#include <string_view>

/*
 * This header contains custom C++ types used to represent the data properties of DAPHNE data objects (e.g., matrices
 * and frames). These types are used in both, the IR/compiler and the runtime.
 */

namespace mlir::daphne {
// EarlyAbort stops growing the distinct-value set after its limit.
// The scan continues to process other requested properties and detect NaNs.
enum class AcquisitionMode : int64_t { Exact = 0, Approx = 1, EarlyAbort = 2 };
inline std::optional<AcquisitionMode> parseAcquisitionMode(std::string_view name) {
    if (name == "exact")
        return AcquisitionMode::Exact;
    if (name == "approx")
        return AcquisitionMode::Approx;
    if (name == "early-abort")
        return AcquisitionMode::EarlyAbort;
    return std::nullopt;
}
enum class BoolOrUnknown { Unknown = -1, False = 0, True = 1 };
} // namespace mlir::daphne

// Make it available in the global namespace for backward compatibility if needed
using mlir::daphne::AcquisitionMode;
using mlir::daphne::BoolOrUnknown;
