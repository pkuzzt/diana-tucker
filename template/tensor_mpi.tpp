#pragma once

#include "util.hpp"
#include "def.hpp"
#include "logger.hpp"
#include "function.hpp"

#include <iostream>
#include <string>
#include <iomanip>
#include <cstdarg>

/*
 * Private member variables.
 */

template<typename Ty>
std::map<Ty *, int> Tensor<Ty>::ref_count = std::map<Ty *, int>();

/*
 * Private member functions.
 */

template<typename Ty>
inline void Tensor<Ty>::init_by_shape(const shape_t &shape) {
    this->op_ = new Operator<Ty>();
    this->ndim_ = shape.size();
    /**
     * Assert that the input shape indicates a tensor.
     * Zero order tensor is a scalar, which is also a kind of tensor.
     **/
    assert(this->ndim_ >= 0);
    this->shape_ = shape_t();
    this->stride_ = shape_t();
    this->stride_in_bytes_ = shape_t();
    this->size_ = 1;
    if (this->ndim_ == 0) {
        return;
    }
    if (this->ndim_ == 2) {
        this->is_matrix_ = true;
        this->trans_ = Transpose::kN;
    }
    for (auto d: shape) {
        this->size_ *= d;
        this->shape_.push_back(d);
        this->stride_.push_back(d);
    }
    // Calculate the stride_ array: column major.
    this->stride_[0] = 1;
    for (size_t i = 1; i < this->ndim_; i++) {
        this->stride_[i] = this->stride_[i - 1] * this->shape_[i - 1];
    }
    // Calculate the stride_in_bytes_ array
    for (auto i: this->stride_) {
        this->stride_in_bytes_.push_back(i);
    }
    for (auto &i: this->stride_in_bytes_) {
        i *= sizeof(Ty);
    }
}

template<typename Ty>
inline void Tensor<Ty>::init_by_distribution(const shape_t &shape,
                                             Distribution *distribution) {
    this->distribution_ = distribution;
    this->shape_global_.assign(shape.begin(), shape.end());
    this->size_global_ = this->distribution_->global_size(this->shape_global_);
    shape_t shape_local;
    this->distribution_->get_local_shape(this->shape_global_, shape_local);
    this->init_by_shape(shape_local);

    this->comm_ = new Communicator<Ty>();
    this->comm_rank_ = this->comm_->rank();
    this->comm_size_ = this->comm_->size();
}

template<typename Ty>
inline void Tensor<Ty>::assert_shape(const Tensor<Ty> &A, const Tensor<Ty> &B) {
#ifndef DIANA_PERFORMANCE_MODE
    assert(A.size() == B.size());
    for (int i = 0; i < A.ndim(); i++) {
        assert(A.shape_[i] == B.shape_[i]);
    }
#endif
}

template<typename Ty>
inline void Tensor<Ty>::assert_permutation(const Tensor<Ty> &A,
                                           const shape_t &perm) {
#ifndef DIANA_PERFORMANCE_MODE
    size_t n = perm.size();
    assert(n == A.ndim());
    shape_t c(n);
    for (auto i: perm) {
        assert(i < n);
        c[i]++;
    }
    for (auto i: c) {
        assert(i == 1);
    }
#endif
}

template<typename Ty>
inline int Tensor<Ty>::shape_t_to_index(const shape_t &idx) {
#ifndef DIANA_PERFORMANCE_MODE
    assert(idx.size() == this->ndim_);
#endif
    int index = 0;
    for (int i = 0; i < this->ndim_; i++) {
#ifndef DIANA_PERFORMANCE_MODE
        assert(idx[i] < this->shape_[i]);
#endif
        index += idx[i] * this->stride_[i];
    }
    return index;
}

/**
 * @brief Construct a new Tensor<Ty>:: Tensor object
 *
 * Do nothing.
 *
 * \warning this->data_, this->op_, this->distribution_ and this->comm_ will be
 * set to nullptr by this constructor.
 *
 * @tparam Ty
 * @param distribution
 */
template<typename Ty>
Tensor<Ty>::Tensor() {
    this->data_ = nullptr;
    this->op_ = nullptr;
    this->distribution_ = nullptr;
    this->comm_ = nullptr;
}

/**
 * @brief Construct a new Tensor< Ty>:: Tensor object
 *
 * \warning this->distribution_ and this->comm_ will be set to nullptr by this
 * constructor.
 *
 * @tparam Ty
 * @param A
 * @param shape global shape.
 */
template<typename Ty>
Tensor<Ty>::Tensor(Ty *A, const shape_t &shape) {
    assert(A != nullptr);
    this->distribution_ = nullptr;
    this->comm_ = nullptr;
    this->init_by_shape(shape);

    this->data_ = A;
}

/**
 * @brief Construct a new local Tensor<Ty>:: Tensor object
 *
 * \warning this->distribution_ and this->comm_ will be set to nullptr by this
 * constructor.
 *
 * @tparam Ty
 * @param shape Tensor shape.
 * @param zero Whether to clear all values to zero (default is true).
 */
template<typename Ty>
Tensor<Ty>::Tensor(const shape_t &shape, bool zero) {
    this->distribution_ = nullptr;
    this->comm_ = nullptr;
    this->init_by_shape(shape);

    this->data_ = this->op_->alloc(this->size_);
    Tensor<Ty>::ref_count[this->data_]++;
    if (zero) {
        this->op_->constant(this->data_, 0, this->size_);
    }
}

/**
 * @brief Construct a new Tensor<Ty>:: Tensor object
 *
 * \warning If distribution->type() is Distribution::Global, user should make
 * sure that datas in all processes are the same.
 *
 * @tparam Ty
 * @param distribution
 * @param A
 * @param shape Global tensor shape.
 */
template<typename Ty>
Tensor<Ty>::Tensor(Distribution *distribution, Ty *A, const shape_t &shape) {
    assert(A != nullptr);
    this->init_by_distribution(shape, distribution);

    this->data_ = A;
    // Do not add ref_count here, A is an external input.
}

/**
 * @brief Construct a new Tensor<Ty>:: Tensor object from a given shape
 *
 * @tparam Ty
 * @param distribution
 * @param shape global shape.
 * @param zero whether to clear all values to zero (default is true).
 */
template<typename Ty>
Tensor<Ty>::Tensor(Distribution *distribution, const shape_t &shape,
                   bool zero) {
    this->init_by_distribution(shape, distribution);

    this->data_ = this->op_->alloc(this->size_);
    Tensor<Ty>::ref_count[this->data_]++;
    if (zero) {
        this->op_->constant(this->data_, 0, this->size_);
    }
}

/**
 * @brief Construct a new Tensor<Ty>:: Tensor object, copy constructor.
 *
 * @tparam Ty
 * @param t
 */
template<typename Ty>
Tensor<Ty>::Tensor(const Tensor<Ty> &t) {
    if (t.distribution() != nullptr) {
        this->init_by_distribution(t.shape_global(), t.distribution());
    } else {
        if (t.data() == nullptr) {
            this->op_ = nullptr;
            this->distribution_ = nullptr;
            this->comm_ = nullptr;
        } else {
            this->distribution_ = nullptr;
            this->comm_ = nullptr;
            this->init_by_shape(t.shape());
        }
    }

    this->data_ = t.data();
    if (this->data_ != nullptr) {
        Tensor<Ty>::ref_count[this->data_]++;
    }
}

/**
 * @brief Destroy the Tensor<Ty>:: Tensor object.
 *
 * @tparam Ty
 */
template<typename Ty>
Tensor<Ty>::~Tensor() {
    delete this->op_;
    delete this->comm_;
    if (this->data_ == nullptr) {
        return;
    }
    if (Tensor<Ty>::ref_count[this->data_] > 0) {
        Tensor<Ty>::ref_count[this->data_]--;
        if (Tensor<Ty>::ref_count[this->data_] == 0) {
            this->op_->free(this->data_);
        }
    }
}

/**
 * @brief Get Tensor<Ty>::data_.
 *
 * @tparam Ty
 * @return Ty* data
 */
template<typename Ty>
inline Ty *Tensor<Ty>::data() const {
    return this->data_;
}

/**
 * @brief Get Tensor<Ty>::op_.
 *
 * @tparam Ty
 * @return Operator<Ty>*
 */
template<typename Ty>
inline Operator<Ty> *Tensor<Ty>::op() const {
    return this->op_;
}

/**
 * @brief Get Tensor<Ty>::ndim_.
 *
 * @tparam Ty
 * @return size_t ndim
 */
template<typename Ty>
inline size_t Tensor<Ty>::ndim() const {
    return this->ndim_;
}

/**
 * @brief Get Tensor<Ty>::size_.
 *
 * @tparam Ty
 * @return size_t size
 */
template<typename Ty>
inline size_t Tensor<Ty>::size() const {
    return this->size_;
}

/**
 * @brief Get Tensor<Ty>::stride_.
 *
 * @tparam Ty
 * @return const shape_t& stride
 */
template<typename Ty>
inline const shape_t &Tensor<Ty>::stride() const {
    return this->stride_;
}

/**
 * @brief Get Tensor<Ty>::stride_in_bytes_.
 *
 * @tparam Ty
 * @return const shape_t& stride_in_bytes
 */
template<typename Ty>
inline const shape_t &Tensor<Ty>::stride_in_bytes() const {
    return this->stride_in_bytes_;
}

/**
 * @brief Get Tensor<Ty>::shape_.
 *
 * @tparam Ty
 * @return const shape_t& shape
 */
template<typename Ty>
inline const shape_t &Tensor<Ty>::shape() const {
    return this->shape_;
}

template<typename Ty>
inline bool Tensor<Ty>::is_matrix() const {
    return this->is_matrix_;
}

template<typename Ty>
inline Transpose Tensor<Ty>::trans() const {
    return this->trans_;
}

template<typename Ty>
inline Distribution *Tensor<Ty>::distribution() const {
    return this->distribution_;
}

template<typename Ty>
inline Communicator<Ty> *Tensor<Ty>::comm() const {
    return this->comm_;
}

template<typename Ty>
inline int Tensor<Ty>::comm_size() const {
    return this->comm_size_;
}

template<typename Ty>
inline int Tensor<Ty>::comm_rank() const {
    return this->comm_rank_;
}

template<typename Ty>
inline const shape_t &Tensor<Ty>::shape_global() const {
    return this->shape_global_;
}

template<typename Ty>
inline size_t Tensor<Ty>::size_global() const {
    return this->size_global_;
}

/**
 * @brief Gather a distributed Tensor in one process to a local Tensor.
 *
 * @tparam Ty
 * @return Tensor<Ty>
 */
template<typename Ty>
Tensor<Ty> Tensor<Ty>::gather() {
    return Function::gather<Ty>(*this);
}

template<typename Ty>
Tensor<Ty> Tensor<Ty>::scatter(Distribution *distribution, int proc) {
    return Function::scatter<Ty>(*this, distribution, proc);
}

template<typename Ty>
void Tensor<Ty>::sync(int proc) {
    this->comm_->bcast(this->data_, (int) this->size_, proc);
}

template<typename Ty>
Ty &Tensor<Ty>::operator[](size_t index) {
    return this->data_[index];
}

template<typename Ty>
Ty &Tensor<Ty>::operator()(size_t x, ...) {
    size_t index = x * this->stride_[0];
    va_list args;
    va_start(args, x);
    for (size_t i = 1; i < this->ndim_; i++) {
        index += va_arg(args, size_t) * this->stride_[i];
    }
    va_end(args);
    return this->data_[index];
}

template<typename Ty>
const Tensor<Ty> Tensor<Ty>::operator=(const Tensor<Ty> &t) {
    if (t.distribution() != nullptr) {
        this->init_by_distribution(t.shape_global(), t.distribution());
    } else {
        this->init_by_shape(t.shape());
    }

    if (this->data_ != nullptr) {
        Tensor<Ty>::ref_count[this->data_]--;
        if (Tensor<Ty>::ref_count[this->data_] == 0) {
            this->op_->free(this->data_);
        }
    }
    this->data_ = t.data();
    if (this->data_ != nullptr) {
        Tensor<Ty>::ref_count[this->data_]++;
    }
    return *this;
}

template<typename Ty>
void Tensor<Ty>::constant(Ty c) {
    this->op_->constant(this->data_, c, this->size_);
}

template<typename Ty>
void Tensor<Ty>::zeros() { this->constant(0); }

template<typename Ty>
void Tensor<Ty>::ones() { this->constant(1); }

template<typename Ty>
void Tensor<Ty>::rand() {
    this->op_->rand(this->data_, this->size_);
}

template<typename Ty>
void Tensor<Ty>::randn() {
    this->op_->randn(this->data_, this->size_);
}

template<typename Ty>
void Tensor<Ty>::add(const Tensor<Ty> &B) {
    Tensor<Ty>::assert_shape(*this, B);
    this->op_->add(this->data_, this->data_, B.data(), this->size_);
}

template<typename Ty>
void Tensor<Ty>::sub(const Tensor<Ty> &B) {
    Tensor<Ty>::assert_shape(*this, B);
    this->op_->sub(this->data_, this->data_, B.data(), this->size_);
}

template<typename Ty>
void Tensor<Ty>::mul(const Tensor<Ty> &B) {
    Tensor<Ty>::assert_shape(*this, B);
    this->op_->nmul(this->data_, this->data_, B.data(), this->size_);
}

template<typename Ty>
void Tensor<Ty>::nmul(Ty c) {
    this->op_->nmul(this->data_, this->data_, c, this->size_);
}

template<typename Ty>
void Tensor<Ty>::ndiv(Ty c) {
    assert(c != 0);
    this->op_->nmul(this->data_, this->data_, 1 / c, this->size_);
}

template<typename Ty>
void Tensor<Ty>::reshape(const shape_t &new_shape) {
    int size_new = 1;
    for (auto i: new_shape) {
        size_new *= i;
    }
    assert(size_new == this->size_);
    this->init_by_distribution(new_shape, this->distribution);
}

template<typename Ty>
void Tensor<Ty>::permute(const shape_t &perm) {
    // TODO: check it!!!
    Tensor<Ty>::assert_permutation(*this, perm);
    auto[data_new, shape_new] =
    this->op_->permute(this->data_, this->shape_, perm);
    // Reinit by shape.
    this->init_by_shape(shape_new);
    // Change data.
    Tensor<Ty>::ref_count[this->data_]--;
    this->op_->free(this->data_);
    this->data_ = data_new;
    Tensor<Ty>::ref_count[this->data_]++;
}

template<typename Ty>
double Tensor<Ty>::fnorm() const {
    return Function::fnorm<Ty>(*this);
}

template<typename Ty>
Ty Tensor<Ty>::sum() const {
    return Function::sum<Ty>(*this);
}

template<typename Ty>
Tensor<Ty> Tensor<Ty>::copy() const {
    if (this->distribution_ != nullptr) {
        Tensor<Ty> ret(this->distribution_, this->shape_global_, false);
        Util::memcpy(ret.data_, this->data_, this->size_ * sizeof(Ty));
        return ret;
    } else {
        Tensor<Ty> ret(this->shape_, false);
        Util::memcpy(ret.data_, this->data_, this->size_ * sizeof(Ty));
        return ret;
    }
}

template<typename Ty>
void Tensor<Ty>::print() const {
    // TODO: Rewrite this function. This implementation is quite ugly.
    std::string ret = "[";
    if (this->ndim_ == 0) {
        ret = std::to_string(this->data_[0]);
    } else if (this->ndim_ == 1) {
        ret += " ";
        for (size_t i = 0; i < this->size_; i++) {
            ret += std::to_string(this->data_[i]);
            if (i != this->size_ - 1) {
                ret += " ";
            }
        }
        ret += "]";
    } else {
        for (size_t i = 0; i < this->size_; i++) {
            bool bracket = false;
            std::string pre = "";
            for (size_t d = 1; d < this->ndim_; d++) {
                if (i % this->stride_[d] == 0) {
                    pre += "[";
                    bracket = true;
                } else if (bracket && i != 0) {
                    pre = " " + pre;
                }
            }
            if (bracket) {
                if (i != 0) {
                    pre = "\n " + pre;
                    if (this->ndim_ > 2 && i % this->stride_[2] == 0) {
                        pre = "\n" + pre;
                    }
                }
                pre += " ";
            }
            ret += pre;
            ret += std::to_string(this->data_[i]);
            ret += " ";
            for (size_t d = 1; d < this->ndim_; d++) {
                if ((i + 1) % this->stride_[d] == 0) {
                    ret += "]";
                }
            }
        }
        ret += "]";
    }
    std::cerr << ret << std::endl;
}

/**
 * Friend functions.
 **/

template<typename Ty>
const Tensor<Ty> operator+(const Tensor<Ty> &A, const Tensor<Ty> &B) {
    Tensor<Ty>::assert_shape(A, B);
    Tensor<Ty> ret(A.shape());
    A.op()->add(ret.data(), A.data(), B.data(), A.size());
    return ret;
}

template<typename Ty>
const Tensor<Ty> operator-(const Tensor<Ty> &A, const Tensor<Ty> &B) {
    Tensor<Ty>::assert_shape(A, B);
    Tensor<Ty> ret(A.shape());
    A.op()->sub(ret.data(), A.data(), B.data(), A.size());
    return ret;
}

template<typename Ty>
const Tensor<Ty> operator*(const Tensor<Ty> &A, const Tensor<Ty> &B) {
    Tensor<Ty>::assert_shape(A, B);
    Tensor<Ty> ret(A.shape());
    A.op()->mul(ret.data(), A.data(), B.data(), A.size());
    return ret;
}

template<typename Ty>
const Tensor<Ty> operator*(Ty c, const Tensor<Ty> &A) {
    Tensor<Ty> ret(A.shape());
    A.op()->nmul(ret.data(), A.data(), c, A.size());
    return ret;
}

template<typename Ty>
const Tensor<Ty> operator*(const Tensor<Ty> &A, Ty c) {
    Tensor<Ty> ret(A.shape());
    A.op()->nmul(ret.data(), A.data(), c, A.size());
    return ret;
}

template<typename Ty>
const Tensor<Ty> operator/(const Tensor<Ty> &A, Ty c) {
    assert(c != 0);
    Tensor<Ty> ret(A.shape());
    A.op()->nmul(ret.data(), A.data(), 1.0 / c, A.size());
    return ret;
}
