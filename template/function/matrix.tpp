#include "tensor.hpp"
#include "logger.hpp"
#include "summary.hpp"

namespace Function {
    template<typename Ty>
    Tensor<Ty> matmulNN(const Tensor<Ty> &A, const Tensor<Ty> &B) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            assert(A.shape()[1] == B.shape()[0]);
            size_t m = A.shape()[0];
            size_t n = B.shape()[1];
            size_t k = A.shape()[1];
            Tensor<Ty> ret({m, n}, false);
            A.op()->matmulNN(ret.data(), A.data(), B.data(), m, n, k);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty> matmulNT(const Tensor<Ty> &A, const Tensor<Ty> &B) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            assert(A.shape()[1] == B.shape()[1]);
            size_t m = A.shape()[0];
            size_t n = B.shape()[0];
            size_t k = A.shape()[1];
            Tensor<Ty> ret({m, n}, false);
            A.op()->matmulNT(ret.data(), A.data(), B.data(), m, n, k);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }


    template<typename Ty>
    Tensor<Ty> matmulTN(const Tensor<Ty> &A, const Tensor<Ty> &B) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            assert(A.shape()[0] == B.shape()[0]);
            size_t m = A.shape()[1];
            size_t n = B.shape()[1];
            size_t k = A.shape()[0];
            Tensor<Ty> ret({m, n}, false);
            A.op()->matmulTN(ret.data(), A.data(), B.data(), m, n, k);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty> inverse(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            assert(A.shape()[0] == A.shape()[1]);
            Tensor<Ty> ret(A.shape(), true);
            A.op()->inverse(ret.data(), A.data(), A.shape()[0]);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_LQ(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            assert(A.shape()[0] <= A.shape()[1]);
            size_t m = A.shape()[0];
            size_t n = A.shape()[1];
            Tensor<Ty> L({m, m}, true);
            Tensor<Ty> Q({m, n}, true);
            A.op()->LQ(L.data(), Q.data(), A.data(), A.shape()[0],
                       A.shape()[1]);
            return std::make_tuple(L, Q);
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    std::tuple<Tensor<Ty>, Tensor<Ty>> reduced_QR(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            assert(A.shape()[0] >= A.shape()[1]);
            size_t m = A.shape()[0];
            size_t n = A.shape()[1];
            Tensor<Ty> Q({m, n}, true);
            Tensor<Ty> R({n, n}, true);
            A.op()->QR(Q.data(), R.data(), A.data(), A.shape()[0],
                       A.shape()[1]);
            return std::make_tuple(Q, R);
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty> transpose(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            size_t m = A.shape()[0];
            size_t n = A.shape()[1];
            Tensor<Ty> At({n, m}, false);
            A.op()->transpose(At.data(), A.data(), m, n);
            return At;
        }
        error("Invalid input or not implemented yet.");
    }


    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr ||
            A.distribution()->type() == Distribution::Type::kGlobal) {
            assert(A.is_matrix());
            size_t M = A.shape()[0];
            size_t N = A.shape()[1];
            Tensor<Ty> ret({M, M}, false);
            A.op()->matmulNT(ret.data(), A.data(), A.data(), M, M, N);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }
}