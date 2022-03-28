#ifndef __DIANA_CORE_INCLUDE_FUNCTION_HPP__
#define __DIANA_CORE_INCLUDE_FUNCTION_HPP__

#include "tensor.hpp"
#include "sp_tensor.hpp"
namespace Function {
    // Matrix functions
    template<typename Ty>
    Tensor<Ty> matmulNN(const Tensor<Ty> &A, const Tensor<Ty> &B);

    template<typename Ty>
    Tensor<Ty> matmulNT(const Tensor<Ty> &A, const Tensor<Ty> &B);

    template<typename Ty>
    Tensor<Ty> matmulTN(const Tensor<Ty> &A, const Tensor<Ty> &B);

    template<typename Ty>
    Tensor<Ty> inverse(const Tensor<Ty> &A);

    template<typename Ty>
    Tensor<Ty> transpose(const Tensor<Ty> &A);

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_LQ(const Tensor<Ty> &A);

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_QR(const Tensor<Ty> &A);

    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A);

    // Tensor functions

    template<typename Ty>
    Tensor<Ty> ttm(const Tensor<Ty> &A, const Tensor<Ty> &M, size_t n);

    template<typename Ty>
    Tensor<Ty>
    ttmc(const Tensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
         const std::vector<size_t> &idx);

    template<typename Ty>
    Tensor<Ty>
    ttmNTc(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
         const std::vector<size_t> &idx, Distribution* distribution, bool to_permu = false);

    template<typename Ty>
    void permutate(Ty* data1, Ty* data2, shape_t & shape, shape_t & permu);

    template<typename Ty>
    void add_outer_product(Ty* data, const size_t *start_index, const size_t *end_index, const size_t *stride,
                           const std::vector<Tensor<Ty>> &M, Ty val, const shape_t &index, size_t dim, const bool * is_contract);

    template<typename Ty>
    void add_outer_product_permu(Ty* data, const size_t *start_index, const size_t *end_index, const size_t *stride,
                           const std::vector<Tensor<Ty>> &M, Ty val, const shape_t &index, size_t dim, const bool * is_contract, shape_t & permu);

    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A, size_t n);


    template<typename Ty>
    Tensor<Ty> gather(const Tensor<Ty> &A);

    template<typename Ty>
    Tensor<Ty>
    scatter(const Tensor<Ty> &A, Distribution *distribution, int proc);

    template<typename Ty>
    double fnorm(const Tensor<Ty> &A);

    template<typename Ty>
    double fnorm(const SpTensor<Ty> &A);

    template<typename Ty>
    Ty sum(const Tensor<Ty> &A);
} // namespace Function


#include "function/matrix.tpp"
#include "function/tensor.tpp"
#include "function/sp_tensor.tpp"
#endif