#ifndef __DIANA_CORE_INCLUDE_ALGORITHM_HPP__
#define __DIANA_CORE_INCLUDE_ALGORITHM_HPP__

#include "tensor.hpp"
#include "sp_tensor.hpp"
#include "def.hpp"
#include "distribution.hpp"
#include <vector>

namespace Algorithm {
    namespace Tucker {
        template<typename Ty>
        std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>
        HOOI_ALS(const Tensor<Ty> &A, const shape_t &R, size_t max_iter);
    }; // namespace GRQI
    namespace SpTucker {
        template<typename Ty>
        std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>
        Sp_HOOI_ALS(const SpTensor<Ty> &A, const shape_t &R, size_t max_iter);
    }
}; // namespace Algorithm

#include "algorithm/tucker/hooi_als.tpp"
#include "algorithm/sp_tucker/sp_hooi_als.tpp"

#endif