#include "util.hpp"

#include <iostream>
#include "summary.hpp"

template<typename Ty>
Ty *Operator<Ty>::alloc(size_t n) {
    return (Ty *) std::malloc(n * sizeof(Ty));
}

template<typename Ty>
void Operator<Ty>::free(Ty *A) { std::free(A); }

template<typename Ty>
void Operator<Ty>::mcpy(Ty *dest, Ty *src, size_t len) {
    Util::memcpy((void *) dest, (void *) src, sizeof(Ty) * len);
}

template<typename Ty>
void Operator<Ty>::transpose(Ty *B, Ty *A, size_t m, size_t n) {
#ifdef DIANA_OPENMP
#pragma omp parallel for default(none) shared(B, A, m, n)
#endif
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            B[j + i * n] = A[i + j * m];
        }
    }
}

template<typename Ty>
void Operator<Ty>::eye(Ty *A, size_t M) {
#ifdef DIANA_OPENMP
#pragma omp parallel for default(none) shared(A, M)
#endif
    for (size_t i = 0; i < M * M; i++) {
        A[i] = 0;
    }
#ifdef DIANA_OPENMP
#pragma omp parallel for default(none) shared(A, M)
#endif
    for (size_t i = 0; i < M; i++) {
        A[i * M + i] = 1;
    }
}

/**
 * @brief Let \f$ \bm{B} = \bm{\mathcal{A}}_{(n)} \f$.
 *
 * @tparam Ty
 * @param B
 * @param A
 * @param shape
 * @param n
 */
template<typename Ty>
void Operator<Ty>::tenmat(Ty *B, Ty *A, const shape_t &shape, size_t n) {
    size_t size = Util::calc_size(shape);
    size_t br = shape[n];            // Row block size of B.
    size_t bc = 1;                   // Column block size of B.
    for (size_t i = 0; i < n; i++) { // Calculate bc.
        bc *= shape[i];
    }
    // Tenmat main process.
    for (size_t idx = 0; idx < size; idx += (br * bc)) {
        for (size_t i = 0; i < bc; i++) {
            for (size_t j = 0; j < br; j++) {
                B[idx + i * br + j] = A[idx + j * bc + i];
            }
        }
    }
}

/**
 * @brief Let \f$ \bm{B} = \bm{\mathcal{A}}_{(n)}^' \f$.
 *
 * @tparam Ty
 * @param B
 * @param A
 * @param shape
 * @param n
 */
template<typename Ty>
void Operator<Ty>::tenmatt(Ty *B, Ty *A, const shape_t &shape, size_t n) {
    // TODO: optimize
    Summary::start(METHOD_NAME);
    size_t size = Util::calc_size(shape);
    size_t br = shape[n];            // Row block size of B.
    size_t bc = 1;                   // Column block size of B.
    for (size_t i = 0; i < n; i++) { // Calculate bc.
        bc *= shape[i];
    }
    size_t col = size / br; // Column size of B;
    // Tenmat main process.
    for (size_t idx = 0; idx < size; idx += (br * bc)) {
        for (size_t j = 0; j < br; j++) {
            for (size_t i = 0; i < bc; i++) {
                B[j * col + idx / br + i] = A[idx + j * bc + i];
            }
        }
    }
    Summary::end(METHOD_NAME);
}

/**
 * @brief Tensorize \f$ \bm{A}_{(n)}^' \f$ and store it in a tensor \f$
 * \bm{\mathcal{B}} \f$.
 *
 * @tparam Ty
 * @param B
 * @param A
 * @param shape
 * @param n
 */
template<typename Ty>
void Operator<Ty>::mattten(Ty *B, Ty *A, const shape_t &shape, size_t n) {
    // TODO: optimize
    Summary::start(METHOD_NAME);
    size_t size = Util::calc_size(shape);
    size_t br = shape[n];            // Row block size of B.
    size_t bc = 1;                   // Column block size of B.
    for (size_t i = 0; i < n; i++) { // Calculate bc.
        bc *= shape[i];
    }
    size_t col = size / br; // Column size of B;
    // Tenmat main process.
    for (size_t idx = 0; idx < size; idx += (br * bc)) {
        for (size_t j = 0; j < br; j++) {
            for (size_t i = 0; i < bc; i++) {
                B[idx + j * bc + i] = A[j * col + idx / br + i];
            }
        }
    }
    Summary::end(METHOD_NAME);
}

template<typename Ty>
Ty Operator<Ty>::sum(Ty *A, size_t n) {
    Ty ret = 0;
    for (size_t i = 0; i < n; i++) {
        ret += A[i];
    }
    return ret;
}