//
// Created by 30250 on 2022/3/17.
//
#include "tensor.hpp"
#include "function.hpp"
#include "logger.hpp"
#include "algorithm.hpp"
#include <tuple>

namespace Algorithm::SpTucker {
    template<typename Ty>
    void distri_orth2(const SpTensor<Ty> &A, Tensor<Ty> &U) {
        
    }

    template<typename Ty>
    void Sp_TTMC_ALS_(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M, size_t n, Tensor<Ty> &R_initial) {
        Summary::start(METHOD_NAME);
        auto L = Function::ttmc_mTR<Ty>(A, M, n, R_initial); // L = not_R_n * R_n

        auto G_norm = Function::fnorm<Ty>(L);
        output("||G||_F = " + std::to_string(G_norm));

        auto G = Function::matmulTN<Ty>(L, L);
        auto G_inv = Function::inverse<Ty>(G); // G_Inv = R_n * R_n
        auto LG_inv = Function::matmulNN<Ty>(L, G_inv);
        Function::ttmc_mTL<Ty>(A, M, n, L, R_initial); // R = R_n * not_R_n
        Operator<Ty>::orth2(R_initial.data(), R_initial.shape()[0], R_initial.shape()[1]);
        Summary::end(METHOD_NAME);
    }

    template<typename Ty>
    std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>

    Sp_HOOI_ALS(const SpTensor<Ty> &A, const shape_t &R, size_t max_iter) {
        const shape_t &I = A.shape();
        const size_t kN = A.ndim();
        auto distribution1 = new DistributionGlobal();
        auto A_norm = Function::fnorm<Ty>(A);
        output("||A||_F = " + std::to_string(A_norm));
        std::vector<Tensor<Ty>> Ut;
        for (size_t n = 0; n < kN; n++) {
            if (n != A.slice_mode) {
                Tensor<double> U_rand(distribution1, {R[n], I[n]}, false);
                U_rand.randn();
                Operator<Ty>::orth2(U_rand.data(), U_rand.shape()[0], U_rand.shape()[1]);
                Ut.push_back(U_rand);
            }
            else {
                Tensor<double> U_rand(distribution1, {R[n], A.slice_end - A.slice_satrt + 1}, false);
                U_rand.randn();
                Algorithm::SpTucker::distri_orth2(A, U_rand);
            }
        }
        for (size_t n = 0; n < kN; n++) {
            Ut[n].sync(0);
        }

        shape_t idx;
        for (size_t i = 0; i < A.ndim(); i++) {
            idx.push_back(i);
        }

        Tensor<Ty> G;
        for (size_t iter = 0; iter < max_iter; iter++) {
            output("Calculating iteration " + std::to_string(iter + 1) +
                   " ...");
            for (size_t n = 0; n < A.ndim(); n++) {
                Algorithm::SpTucker::Sp_TTMC_ALS_(A, Ut, n, Ut[n]);
            }
        }
        G = Function::ttmc_mTR(A, Ut, A.ndim() - 1, Ut[A.ndim() - 1]);
        auto G_norm = Function::fnorm<Ty>(G);

        output("||G||_F = " + std::to_string(G_norm));
        output("Residual: sqrt(1 - ||G||_F^2 / ||A||_F^2) = " +
               std::to_string(
                       sqrt(1 - (G_norm * G_norm) / (A_norm * A_norm))));
        return std::make_tuple(G, Ut);
    }
}
