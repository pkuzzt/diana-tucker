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
    std::tuple<Tensor<Ty>, std::vector<Tensor<Ty>>>
    Sp_HOOI_ALS(const SpTensor<Ty> &A, const shape_t &R, size_t max_iter, Distribution *distribution) {
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

        Tensor<Ty> G;
        for (size_t iter = 0; iter < max_iter; iter++) {
            output("Calculating iteration " + std::to_string(iter + 1) +
                   " ...");
            for (size_t n = 0; n < A.ndim(); n++) {
                shape_t idx;
                for (size_t j = 0; j < A.ndim(); j++) {
                    if (j != n) {
                        idx.push_back(j);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                auto Y = Function::ttmNTc(A, U, idx, distribution, true); //TODO:METTM
                MPI_Barrier(MPI_COMM_WORLD);
                U[n] = Algorithm::Tucker::ALS_(Y, n, U[n]);
                MPI_Barrier(MPI_COMM_WORLD);
                if (n == A.ndim() - 1) {
                    auto Ut = Function::transpose<Ty>(U[n]);
                    G = Function::ttm<Ty>(Y, Ut, n);
                    auto G_norm = Function::fnorm<Ty>(G);
                    output("||G||_F = " + std::to_string(G_norm));
                    output("Residual: sqrt(1 - ||G||_F^2 / ||A||_F^2) = " +
                           std::to_string(
                                   sqrt(1 - (G_norm * G_norm) / (A_norm * A_norm))));
                }
            }
        }
        return std::make_tuple(G, U);
    }
}
