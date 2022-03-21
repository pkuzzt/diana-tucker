#include "operator.hpp"
#include "util.hpp"
#include "logger.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstdlib>

#ifdef DIANA_MKL
extern "C" {
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
}
#else
#ifdef DIANA_BLAS
extern "C" {
#include "cblas.h"
#include "lapacke.h"
}
#endif
#endif

template<>
void Operator<double>::add(double *C, double *A, double *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

template<>
void Operator<double>::sub(double *C, double *A, double *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

template<>
void Operator<double>::mul(double *C, double *A, double *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

template<>
void Operator<double>::nmul(double *C, double *A, double B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B;
    }
}

template<>
void Operator<double>::constant(double *A, double c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = c;
    }
}

template<>
void Operator<double>::rand(double *A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = (double) std::rand() / RAND_MAX;
    }
}

template<>
void Operator<double>::randn(double *A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = Util::randn();
    }
}

template<>
double Operator<double>::fnorm(double *A, size_t n) {
    double ret = 0;
    for (size_t i = 0; i < n; i++) {
        ret += A[i] * A[i];
    }
    return std::sqrt(ret);
}

template<>
void
Operator<double>::inverse(double *C, double *A, size_t m) {
#ifdef DIANA_LAPACK
    Summary::start(METHOD_NAME);
    auto M = (lapack_int) m;
    lapack_int LDA = M;
    lapack_int INFO;
    auto IPIV = Operator<lapack_int>::alloc(m);
    Operator<double>::mcpy(C, A, m * m);
    INFO = LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, M, C, LDA, IPIV);
    checkwarn(INFO == 0);
    INFO = LAPACKE_dgetri(LAPACK_COL_MAJOR, M, C, LDA, IPIV);
    checkwarn(INFO == 0);
    Operator<lapack_int>::free(IPIV);
    Summary::end(METHOD_NAME);
#else
    fatal("Cannot calculate inverse without BLAS!");
#endif
}

template<>
void Operator<double>::LQ(double *L, double *Q, double *A, size_t m, size_t n) {
#ifdef DIANA_LAPACK
    Summary::start(METHOD_NAME);
    auto M = (lapack_int) m;
    auto N = (lapack_int) n;
    lapack_int LDA = M;
    lapack_int INFO;
    double *TAU = Operator<double>::alloc(std::min(m, n));
    Operator<double>::mcpy(Q, A, m * n);
    INFO = LAPACKE_dgelqf(LAPACK_COL_MAJOR, M, N, Q, LDA, TAU);
    checkwarn(INFO == 0);
    char UPLO = 'L';
    lapack_int LDB = M;
    INFO = LAPACKE_dlacpy(LAPACK_COL_MAJOR, UPLO, M, N, Q, LDA, L, LDB);
    checkwarn(INFO == 0);
    lapack_int K = std::min(M, N);
    INFO = LAPACKE_dorglq(LAPACK_COL_MAJOR, M, N, K, Q, LDA, TAU);
    checkwarn(INFO == 0);
    Operator<double>::free(TAU);
    Summary::end(METHOD_NAME);
#else
    fatal("Cannot calculate inverse without BLAS!");
#endif
}

template<>
void Operator<double>::QR(double *Q, double *R, double *A, size_t m, size_t n) {
#ifdef DIANA_LAPACK
    Summary::start(METHOD_NAME);
    auto M = (lapack_int) m;
    auto N = (lapack_int) n;
    lapack_int LDA = M;
    lapack_int INFO;
    double *TAU = Operator<double>::alloc(std::min(m, n));
    Operator<double>::mcpy(Q, A, m * n);
    INFO = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, M, N, Q, LDA, TAU);
    checkwarn(INFO == 0);
    char UPLO = 'U';
    lapack_int LDB = N;
    INFO = LAPACKE_dlacpy(LAPACK_COL_MAJOR, UPLO, M, N, Q, LDA, R, LDB);
    checkwarn(INFO == 0);
    lapack_int K = std::min(M, N);
    INFO = LAPACKE_dorgqr(LAPACK_COL_MAJOR, M, N, K, Q, LDA, TAU);
    checkwarn(INFO == 0);
    Operator<double>::free(TAU);
    Summary::end(METHOD_NAME);
#else
    fatal("Cannot calculate inverse without BLAS!");
#endif
}

template<>
void Operator<double>::matmulNN(double *C, double *A, double *B, size_t m,
                                size_t n, size_t k) {
#ifdef DIANA_BLAS
    Summary::start(METHOD_NAME, 2 * (long long) m * (long long) n *
                                (long long) k);
    double alpha = 1.0;
    int lda = (int) m;
    int ldb = (int) k;
    double beta = 0.0;
    int ldc = (int) m;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) n,
                (int) k, alpha, A, lda, B, ldb, beta, C, ldc);
    Summary::end(METHOD_NAME);
#else
    fatal("Cannot calculate matmulNN without BLAS!");
#endif
}

template<>
void Operator<double>::matmulNT(double *C, double *A, double *B, size_t m,
                                size_t n, size_t k) {
#ifdef DIANA_BLAS
    Summary::start(METHOD_NAME, 2 * (long long) m * (long long) n *
                                (long long) k);
    double alpha = 1.0;
    int lda = (int) m;
    int ldb = (int) n;
    double beta = 0.0;
    int ldc = (int) m;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, (int) m, (int) n,
                (int) k,
                alpha, A, lda, B, ldb, beta, C, ldc);
    Summary::end(METHOD_NAME);
#else
    fatal("Cannot calculate matmulNT without BLAS!");
#endif
}


template<>
void Operator<double>::matmulTN(double *C, double *A, double *B, size_t m,
                                size_t n, size_t k) {
#ifdef DIANA_BLAS
    Summary::start(METHOD_NAME, 2 * (long long) m * (long long) n *
                                (long long) k);
    double alpha = 1.0;
    int lda = (int) k;
    int ldb = (int) k;
    double beta = 0.0;
    int ldc = (int) m;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) m, (int) n,
                (int) k,
                alpha, A, lda, B, ldb, beta, C, ldc);
    Summary::end(METHOD_NAME);
#else
    fatal("Cannot calculate matmulNT without BLAS!");
#endif
}
