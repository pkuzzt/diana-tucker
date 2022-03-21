#include "logger.hpp"

template<typename Ty>
void Operator<Ty>::reorder_from_gather_cartesian_block(Ty *A,
                                                       const shape_t &shape,
                                                       const shape_t &partition,
                                                       int *displs) {
    Summary::start(METHOD_NAME);
    const size_t kSize = Util::calc_size(shape);
    const size_t kNdim = shape.size();
    const size_t kMPISize = Util::calc_size(partition);
    Ty *B = Operator<Ty>::alloc(kSize);
    Operator<Ty>::mcpy(B, A, kSize);
    int *displs_ = Operator<int>::alloc(kMPISize);
    Operator<int>::mcpy(displs_, displs, kMPISize);
    // Enumerate items in A, and place data from B to A.
    for (size_t i = 0; i < kSize; i++) {
        size_t rank = 0;
        size_t j = i;
        size_t pre = 1;
        for (size_t d = 0; d < kNdim; d++) {
            const size_t kJ = j % shape[d];
            const size_t kD = shape[d] / partition[d];
            const size_t kM = shape[d] % partition[d];
            j /= shape[d];
            size_t p;
            if (kJ < kM * (kD + 1)) {
                p = kJ / (kD + 1);
            } else {
                p = (kJ - kM * (kD + 1)) / kD + kM;
            }
            rank += p * pre;
            pre *= partition[d];
        }
        A[i] = B[displs_[rank]++];
    }
    Operator<Ty>::free(B);
    Operator<int>::free(displs_);
    Summary::end(METHOD_NAME);
}

template<typename Ty>
void Operator<Ty>::reorder_for_scatter_cartesian_block(Ty *A,
                                                       const shape_t &shape,
                                                       const shape_t &partition,
                                                       int *displs) {
    Summary::start(METHOD_NAME);
    const size_t kSize = Util::calc_size(shape);
    const size_t kNdim = shape.size();
    const size_t kMPISize = Util::calc_size(partition);
    Ty *B = Operator<Ty>::alloc(kSize);
    Operator<Ty>::mcpy(B, A, kSize);
    int *displs_ = Operator<int>::alloc(kMPISize);
    Operator<int>::mcpy(displs_, displs, kMPISize);
    // Enumerate items in B, and place data from B to A.
    for (size_t i = 0; i < kSize; i++) {
        size_t rank = 0;
        size_t j = i;
        size_t pre = 1;
        for (size_t d = 0; d < kNdim; d++) {
            const size_t kJ = j % shape[d];
            const size_t kD = shape[d] / partition[d];
            const size_t kM = shape[d] % partition[d];
            j /= shape[d];
            size_t p;
            if (kJ < kM * (kD + 1)) {
                p = kJ / (kD + 1);
            } else {
                p = (kJ - kM * (kD + 1)) / kD + kM;
            }
            rank += p * pre;
            pre *= partition[d];
        }
        A[displs_[rank]++] = B[i];
    }
    Operator<Ty>::free(B);
    Operator<int>::free(displs_);
    Summary::end(METHOD_NAME);
}