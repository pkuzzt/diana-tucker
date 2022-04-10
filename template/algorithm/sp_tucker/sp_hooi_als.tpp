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
    Tensor<Ty> Sp_TTMC_ALS_(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M, size_t n, const Tensor<Ty> &R_initial) {
        Summary::start(METHOD_NAME);
        auto L = Function::ttmc_mTR<Ty>(A, M, n, R_initial); // L = not_R_n * R_n
        auto G = Function::matmulTN<Ty>(L, L);
        auto G_inv = Function::inverse<Ty>(G); // G_Inv = R_n * R_n
        auto LG_inv = Function::matmulNN<Ty>(L, G_inv);
        auto R = Function::ttmc_mTL<Ty>(A, M, n, L); // R = R_n * not_R_n
        auto tr_R = Function::transpose<Ty>(R);
        auto[q, r] = Function::reduced_QR<Ty>(tr_R);
        Summary::end(METHOD_NAME);
        return q;
    }

    template<typename Ty>
    std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>

    Sp_HOOI_ALS(const SpTensor<Ty> &A, const shape_t &R, size_t max_iter) {
        const shape_t &I = A.shape();
        const size_t kN = A.ndim();
        auto distribution1 = new DistributionGlobal();
        auto A_norm = Function::fnorm<Ty>(A);
        output("||A||_F = " + std::to_string(A_norm));
        std::vector<Tensor<Ty>> U;
        for (size_t n = 0; n < kN; n++) {
            Tensor<double> U_rand(distribution1, {I[n], R[n]}, false);
            U_rand.randn();
            auto[q, r] = Function::reduced_QR<Ty>(U_rand);
            U.push_back(q);
        }
        for (size_t n = 0; n < kN; n++) {
            U[n].sync(0);
        }

        std::vector<Tensor<Ty>> Ut;
        for (size_t n = 0; n < kN; n++) {
            auto Ut_n = Function::transpose<Ty>(U[n]);
            Ut.push_back(Ut_n);
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
//                for (size_t t = 0; t < 2; t++) {
//                    U[n] = Algorithm::SpTucker::Sp_TTMC_ALS_(A, Ut, n, Ut[n]);
//                    Ut[n] = Function::transpose<Ty>(U[n]);
//                }
                U[n] = Algorithm::SpTucker::Sp_TTMC_ALS_(A, Ut, n, Ut[n]);
                Ut[n] = Function::transpose<Ty>(U[n]);
                if (n == A.ndim() - 1) {

                }
            }
        }
        G = Function::ttmc_mTR(A, Ut, A.ndim() - 1, Ut[A.ndim() - 1]);
        auto G_norm = Function::fnorm<Ty>(G);
        output("||G||_F = " + std::to_string(G_norm));
        output("Residual: sqrt(1 - ||G||_F^2 / ||A||_F^2) = " +
               std::to_string(
                       sqrt(1 - (G_norm * G_norm) / (A_norm * A_norm))));
        return std::make_tuple(G, U);
    }
}
