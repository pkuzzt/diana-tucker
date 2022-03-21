#include "quantum/qregister.hpp"

#include "tensor.hpp"
#include "logger.hpp"
#include "util.hpp"
#include <cmath>

SubspaceCPQRegister::SubspaceCPQRegister(int qubit_number, vint par)
        : QRegister(qubit_number) {
    int size_assert = 0;
    for (auto i: par) {
        size_assert += i;
    }
    assert(size_assert == qubit_number);

    this->par_.assign(par.begin(), par.end());
    this->p_ = par.size();
    this->rank_ = 1;

    vint shape;
    int j = 0;
    int n = this->qubit_number();
    for (int i = 0; i < n; i++) {
        shape.push_back(2);
        if (shape.size() == par[j]) {
            shape.push_back(1);
            this->factor_.push_back(
                    Tensor<complex64>(shape, true, Distribution::kBlock1D));
            j++;
            shape = vint();
        }
    }
    if (this->comm_.rank() == 0) {
        // TODO: change this part to kronecker product.
        for (int i = 0; i < this->p_; i++) {
            this->factor_[i][0] = 1;
        }
    }

    this->comm_rank_ = this->comm_.rank();
    this->comm_size_ = this->comm_.size();
}

SubspaceCPQRegister::~SubspaceCPQRegister() {}

std::vector<Tensor<complex64>> SubspaceCPQRegister::generate_p() {
    std::vector<Tensor<complex64>> ret;
    for (int i = 0; i < 4; i++) {
        Tensor<complex64> p({2, 2});
        p[i] = 1;
        ret.push_back(p);
    }
    return ret;
}

std::vector<Tensor<complex64>>
SubspaceCPQRegister::partition(Tensor<complex64> w) {
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

std::tuple<int, int> SubspaceCPQRegister::get_index(int x) {
    for (int i = 0; i < this->p_; i++) {
        if (x >= this->par_[i]) {
            x -= this->par_[i];
        } else {
            return std::make_tuple(i, x);
        }
    }
}

complex64 SubspaceCPQRegister::amplitude(long long index) {
    Tensor<complex64> ret({this->rank_}, false);
    ret.ones();
    for (int i = 0; i < this->p_; i++) {
        int factor_index = index & ((1 << this->par_[i]) - 1);
        index >>= this->par_[i];
        int local_size = this->factor_[i].size();
        int rank = factor_index / local_size;
        Tensor<complex64> fac({this->rank_}, false);
        if (rank == this->comm_rank_) {
            Util::memcpy(fac.data(),
                         this->factor_[i].data() +
                         (factor_index % local_size) * this->rank_,
                         this->rank_);
        }
        this->comm_.bcast(fac.data(), this->rank_, rank);
        ret.mul(fac);
    }
    ret.mul(this->lambda_);
    return ret.sum();
}

void SubspaceCPQRegister::apply_single_gate(const Gate &g) {
    int x = g.qubit_id()[0];
    auto[fac, idx] = get_index(x);
    ttm_2x2_inplace_with_r(this->factor_[fac], g.weight(), idx, this->rank_);
}

void SubspaceCPQRegister::apply_controlled_gate(const Gate &g) {
    // TODO: extremely fucking ugly.
    int x = g.qubit_id()[0];
    int y = g.qubit_id()[1];
    auto p = generate_p();
    auto q = partition(g.weight());
    auto[fac_x, idx_x] = get_index(x);
    auto[fac_y, idx_y] = get_index(x);
    if (fac_x == fac_y) {
        auto sv_copy = this->factor_[fac_x].copy();
        ttm_2x2_inplace_with_r(this->factor_[fac_x], p[0], idx_x, this->rank_);
        ttm_2x2_inplace_with_r(this->factor_[fac_x], q[0], idx_y, this->rank_);
        ttm_2x2_inplace_with_r(sv_copy, p[3], idx_x, this->rank_);
        ttm_2x2_inplace_with_r(sv_copy, q[3], idx_y, this->rank_);
        this->factor_[fac_x].add(sv_copy);
    } else {
        this->factor_changed_idx_[0] = fac_x;
        this->factor_changed_idx_[1] = fac_y;
        this->factor_changed_len_[0] = 2;
        this->factor_changed_len_[1] = 2;
        // factor x.
        this->factor_changed_[0][0] = this->factor_[fac_x].copy();
        ttm_2x2_inplace_with_r(this->factor_changed_[0][0], p[0], idx_x,
                               this->rank_);
        this->factor_changed_[0][1] = this->factor_[fac_x].copy();
        ttm_2x2_inplace_with_r(this->factor_changed_[0][1], p[3], idx_x,
                               this->rank_);
        // factor y.
        this->factor_changed_[1][0] = this->factor_[fac_y].copy();
        ttm_2x2_inplace_with_r(this->factor_changed_[1][0], q[0], idx_y,
                               this->rank_);
        this->factor_changed_[1][1] = this->factor_[fac_y].copy();
        ttm_2x2_inplace_with_r(this->factor_changed_[1][1], q[3], idx_y,
                               this->rank_);
    }
}

void SubspaceCPQRegister::apply_double_gate(const Gate &g) {
    // TODO: fucking ugly.
    int x = g.qubit_id()[0];
    int y = g.qubit_id()[1];
    auto p = generate_p();
    auto q = partition(g.weight());
    auto[fac_x, idx_x] = get_index(x);
    auto[fac_y, idx_y] = get_index(x);
    if (fac_x == fac_y) {
        auto sv_copy1 = this->factor_[fac_x].copy();
        auto sv_copy2 = this->factor_[fac_x].copy();
        auto sv_copy3 = this->factor_[fac_x].copy();
        ttm_2x2_inplace_with_r(this->factor_[fac_x], p[0], x, this->rank_);
        ttm_2x2_inplace_with_r(this->factor_[fac_x], q[0], y, this->rank_);
        ttm_2x2_inplace_with_r(sv_copy1, p[1], idx_x, this->rank_);
        ttm_2x2_inplace_with_r(sv_copy1, q[1], idx_y, this->rank_);
        this->factor_[fac_x].add(sv_copy1);
        ttm_2x2_inplace_with_r(sv_copy2, p[2], idx_x, this->rank_);
        ttm_2x2_inplace_with_r(sv_copy2, q[2], idx_y, this->rank_);
        this->factor_[fac_x].add(sv_copy2);
        ttm_2x2_inplace_with_r(sv_copy3, p[3], idx_x, this->rank_);
        ttm_2x2_inplace_with_r(sv_copy3, q[3], idx_y, this->rank_);
        this->factor_[fac_x].add(sv_copy3);
    } else {
        this->factor_changed_idx_[0] = fac_x;
        this->factor_changed_idx_[1] = fac_y;
        this->factor_changed_len_[0] = 4;
        this->factor_changed_len_[1] = 4;
        for (int i = 0; i < 4; i++) {
            this->factor_changed_[0][i] = this->factor_[fac_x].copy();
            ttm_2x2_inplace_with_r(this->factor_changed_[0][i], p[i], idx_x,
                                   this->rank_);
        }
        for (int i = 0; i < 4; i++) {
            this->factor_changed_[1][i] = this->factor_[fac_y].copy();
            ttm_2x2_inplace_with_r(this->factor_changed_[1][i], q[i], idx_y,
                                   this->rank_);
        }
    }
}

int update_rank_(int rank) { return rank + 1; }

void SubspaceCPQRegister::compress(float64 fidelity) {
    const float64 EPS = 1e-4;
    std::vector<Tensor<complex64>> U_new;
    std::vector<vint> shape_ori;
    int rank_now = (this->rank_ / 4 < 1) ? 1 : (this->rank_ / 4);
    float64 fnorm_initial = 0;

    for (int i = 0; i < this->p_; i++) {
        shape_ori.push_back(this->factor_[i].shape());
        vint shape_new{this->factor_[i].size() / this->rank_, this->rank_};
        this->factor_[i].reshape(shape_new);
        fnorm_initial += this->factor_[i].fnorm() * this->factor_[i].fnorm();
        Tensor<complex64> factor_new(vint{shape_new[0], rank_now}, false);
        factor_new.randn();
        U_new.push_back(factor_new);
    }
    while (true) {
        float64 fnorm_now = 0;
        float64 fnorm_pre = 0;
        while (true) {
            // ALS.
            fnorm_now = 0;
            for (int k = 0; k < this->p_; k++) {
                Tensor<complex64> H1, H2;
                for (int i = 0; i < this->p_; i++) {
                    if (i == k) {
                        continue;
                    }
                    if (H1.data() == nullptr && H2.data() == nullptr) {
                        H1 = matmul(U_new[i].T(), U_new[i]);
                        H2 = matmul(this->factor_[i].T(), U_new[i]);
                    } else {
                        H1.mul(matmul(U_new[i].T(), U_new[i]));
                        H2.mul(matmul(this->factor_[i].T(), U_new[i]));
                    }
                }
                H1 = inv(H1);
                U_new[k] = matmul(matmul(this->factor_[k], H2), H1);
                fnorm_now += U_new[k].fnorm() * U_new[k].fnorm();
            }
            // Fidelity measure.
            if (fabs((fnorm_now - fnorm_pre) / fnorm_initial) < EPS) {
                break;
            } else {
                fnorm_pre = fnorm_now;
            }
        }
        // Fidelity measure。
        if (fabs(1 - fnorm_now) > fidelity) {
            break;
        }
        // Update rank.
        int rank_pre = rank_now;
        rank_now = update_rank_(rank_now);
        for (int i = 0; i < this->p_; i++) {
            int m = U_new[i].shape()[0];
            int n = U_new[i].shape()[1];
            Tensor<complex64> factor_new(vint{m, rank_now});
            complex64 *data_f = factor_new.data();
            complex64 *data_u = U_new[i].data();
            for (int j = 0; j < m; j++) {
                Util::memcpy(data_f + j * rank_now, data_u + j * rank_pre,
                             rank_pre * sizeof(complex64));
                Operator<complex64>::randn(data_f + j * rank_now + rank_pre,
                                           rank_now - rank_pre);
            }
            U_new[i] = factor_new;
        }
    }
    for (int i = 0; i < this->p_; i++) {
        shape_ori.push_back(this->factor_[i].shape());
        this->factor_[i].reshape(shape_ori[i]);
    }
}

float64 SubspaceCPQRegister::probability(long long index) {
    complex64 amp = this->amplitude(index);
    return amp.real() * amp.real() + amp.imag() * amp.imag();
}

void SubspaceCPQRegister::apply_gate(const Gate &g) {
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

void SubspaceCPQRegister::apply_circuit(const Circuit &c, float64 fidelity) {
    for (auto g: c.gate) {
        apply_gate(g);
        compress(fidelity);
    }
}

void SubspaceCPQRegister::print() {
    for (int fac = 0; fac < this->p_; fac++) {
        if (this->comm_rank_ == 0) {
            info("Factor: " + std::to_string(fac));
        }
        for (int i = 0; i < this->comm_size_; i++) {
            if (i == this->comm_rank_) {
                this->factor_[fac].print();
            }
            this->comm_.barrier();
        }
    }
}