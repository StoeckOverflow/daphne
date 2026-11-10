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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ANALYZEDATA_H
#define SRC_RUNTIME_LOCAL_KERNELS_ANALYZEDATA_H

#include <runtime/local/kernels/AnalyzeProperties.h>

// Adapt the public operation/kernel ABI to an internal acquisition request.
template <typename DT>
void analyzeData(const DT *arg, bool analyzeSparsity, bool analyzeSymmetric, bool analyzeSortness, bool analyzeMinMax,
                 bool analyzeDistinct, bool analyzeSparsityPattern, DCTX(ctx)) {
    AnalysisRequest request{analyzeSparsity, analyzeSymmetric, analyzeSortness,       analyzeMinMax,
                            analyzeMinMax,   analyzeDistinct,  analyzeSparsityPattern};
    AnalyzeProperties<DT>::apply(arg, request, ctx);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_ANALYZEDATA_H
