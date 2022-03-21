#include "operator.hpp"
#include "util.hpp"

#include <cstdlib>

template <> void Operator<float>::add(float *C, float *A, float *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

template <> void Operator<float>::sub(float *C, float *A, float *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

template <> void Operator<float>::mul(float *C, float *A, float *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

template <> void Operator<float>::constant(float *A, float c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = c;
    }
}

template <> void Operator<float>::randn(float *A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = (float)Util::randn();
    }
}