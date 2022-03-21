#include "operator.hpp"
#include "util.hpp"
#include "logger.hpp"
#include "def.hpp"

#include <cstdlib>
#include <cmath>

template<>
void Operator<complex64>::add(complex64 *C, complex64 *A, complex64 *B,
                              size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

template<>
void Operator<complex64>::sub(complex64 *C, complex64 *A, complex64 *B,
                              size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

template<>
void Operator<complex64>::mul(complex64 *C, complex64 *A, complex64 *B,
                              size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

template<>
void Operator<complex64>::nmul(complex64 *C, complex64 *A, complex64 B,
                               size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B;
    }
}

template<>
void Operator<complex64>::constant(complex64 *A, complex64 c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = c;
    }
}

template<>
void Operator<complex64>::randn(complex64 *A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = Util::randn() + Util::randn() * complex64(0, 1);
    }
}

template<>
double Operator<complex64>::fnorm(complex64 *A, size_t n) {
    double ret = 0;
    for (size_t i = 0; i < n; i++) {
        ret += A[i].real() * A[i].real() + A[i].imag() * A[i].imag();
    }
    return std::sqrt(ret);
}