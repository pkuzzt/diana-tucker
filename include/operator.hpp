#ifndef __DIANA_CORE_SRC_INCLUDE_OPERATOR_HPP__
#define __DIANA_CORE_SRC_INCLUDE_OPERATOR_HPP__

#include "def.hpp"

#include <tuple>

template<typename Ty>
class Operator {
public:
    static Ty *alloc(size_t);

    static void free(Ty *);

    static void mcpy(Ty *dest, Ty *src, size_t len);

    static void add(Ty *, Ty *, Ty *, size_t);

    static void sub(Ty *, Ty *, Ty *, size_t);

    static void mul(Ty *, Ty *, Ty *, size_t);

    static void nmul(Ty *, Ty *, Ty, size_t);

    static void eye(Ty *, size_t);

    static void constant(Ty *, Ty, size_t);

    static void rand(Ty *, size_t);

    static void randn(Ty *, size_t);

    static void inverse(Ty *C, Ty *A, size_t n);

    static void LQ(Ty *L, Ty *Q, Ty *A, size_t m, size_t n);

    static void QR(Ty *Q, Ty *R, Ty *A, size_t m, size_t n);

    static void matmulNN(Ty *C, Ty *A, Ty *B, size_t m, size_t n, size_t k);

    static void matmulNT(Ty *C, Ty *A, Ty *B, size_t m, size_t n, size_t k);

    static void matmulTN(Ty *C, Ty *A, Ty *B, size_t m, size_t n, size_t k);

    static void transpose(Ty *B, Ty *A, size_t m, size_t n);

    static void tenmat(Ty *B, Ty *A, const shape_t &shape, size_t n);

    static void tenmatt(Ty *B, Ty *A, const shape_t &shape, size_t n);

    static void mattten(Ty *B, Ty *A, const shape_t &shape, size_t n);

    static double fnorm(Ty *, size_t);

    static Ty sum(Ty *, size_t);

    static void reorder_from_gather_cartesian_block(Ty *A, const shape_t &shape,
                                                    const shape_t &partition,
                                                    int *displs);

    static void reorder_for_scatter_cartesian_block(Ty *A, const shape_t &shape,
                                                    const shape_t &partition,
                                                    int *displs);
};

template
class Operator<double>;

template
class Operator<float>;

template
class Operator<complex32>;

template
class Operator<complex64>;

#include "operator/operator_cpu.tpp"
#include "operator/operator_cpu_cartesian_block.tpp"

#endif