#include "quantum/qregister.hpp"

#include "tensor.hpp"
#include "logger.hpp"
#include <cmath>

SchrodingerQRegister::SchrodingerQRegister(int qubit_number)
    : QRegister(qubit_number) {
    vint shape;
    int n = this->qubit_number();
    for (int i = 0; i < n; i++) {
        shape.push_back(2);
    }
    this->state_vector_ =
        Tensor<complex64>(shape, true, Distribution::kBlock1D);
    if (this->comm_.rank() == 0) {
        this->state_vector_[0] = 1;
    }

    this->comm_rank_ = this->comm_.rank();
    this->comm_size_ = this->comm_.size();
}

SchrodingerQRegister::~SchrodingerQRegister() {}

std::vector<Tensor<complex64>> SchrodingerQRegister::generate_p() {
    std::vector<Tensor<complex64>> ret;
    for (int i = 0; i < 4; i++) {
        Tensor<complex64> p({2, 2});
        p[i] = 1;
        ret.push_back(p);
    }
    return ret;
}

std::vector<Tensor<complex64>>
SchrodingerQRegister::partition(Tensor<complex64> w) {
    std::vector<Tensor<complex64>> ret;
    for (int k = 0; k < 4; k++) {
        Tensor<complex64> q({2, 2});
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                q[j * 2 + i] = w[(j + 2 * (k / 2)) * 4 + (i + 2 * (k % 2))];
            }
        }
        ret.push_back(q);
    }
    return ret;
}

complex64 SchrodingerQRegister::amplitude(long long index) {
    int local_size = this->state_vector_.size();
    int rank = index / local_size;
    complex64 ret;
    if (rank == this->comm_rank_) {
        ret = this->state_vector_[index % local_size];
    }
    this->comm_.bcast(&ret, 1, rank);
    return ret;
}

void SchrodingerQRegister::apply_single_gate(const Gate &g) {
    int x = g.qubit_id()[0];
    ttm_2x2_inplace(this->state_vector_, g.weight(), x);
}

void SchrodingerQRegister::apply_controlled_gate(const Gate &g) {
    // TODO: fucking ugly.
    int x = g.qubit_id()[0];
    int y = g.qubit_id()[1];
    auto p = generate_p();
    auto q = partition(g.weight());
    auto sv3 = this->state_vector_.copy();
    ttm_2x2_inplace(this->state_vector_, p[0], x);
    ttm_2x2_inplace(this->state_vector_, q[0], y);
    ttm_2x2_inplace(sv3, p[3], x);
    ttm_2x2_inplace(sv3, q[3], y);
    this->state_vector_.add(sv3);
}

void SchrodingerQRegister::apply_double_gate(const Gate &g) {
    // TODO: fucking ugly.
    int x = g.qubit_id()[0];
    int y = g.qubit_id()[1];
    auto p = generate_p();
    auto q = partition(g.weight());
    auto sv1 = this->state_vector_.copy();
    auto sv2 = this->state_vector_.copy();
    auto sv3 = this->state_vector_.copy();
    ttm_2x2_inplace(this->state_vector_, p[0], x);
    ttm_2x2_inplace(this->state_vector_, q[0], y);
    ttm_2x2_inplace(sv1, p[1], x);
    ttm_2x2_inplace(sv1, q[1], y);
    this->state_vector_.add(sv1);
    ttm_2x2_inplace(sv2, p[2], x);
    ttm_2x2_inplace(sv2, q[2], y);
    this->state_vector_.add(sv2);
    ttm_2x2_inplace(sv3, p[3], x);
    ttm_2x2_inplace(sv3, q[3], y);
    this->state_vector_.add(sv3);
}

float64 SchrodingerQRegister::probability(long long index) {
    complex64 amp = this->amplitude(index);
    return amp.real() * amp.real() + amp.imag() * amp.imag();
}

void SchrodingerQRegister::apply_gate(const Gate &g) {
    if (g.type() == GateType::kSingle) {
        apply_single_gate(g);
        return;
    }
    if (g.type() == GateType::kControlled) {
        apply_controlled_gate(g);
        return;
    }
    if (g.type() == GateType::kDouble || g.type() == GateType::kSwap) {
        apply_double_gate(g);
        return;
    }
}

void SchrodingerQRegister::apply_circuit(const Circuit &c) {
    for (auto g : c.gate) {
        apply_gate(g);
    }
}

void SchrodingerQRegister::print() {
    for (int i = 0; i < this->comm_size_; i++) {
        if (i == this->comm_rank_) {
            this->state_vector_.print();
        }
        this->comm_.barrier();
    }
}