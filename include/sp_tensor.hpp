//
// Created by 30250 on 2022/3/17.
//

#ifndef DIANA_TUCKER_SP_TENSOR_HPP
#define DIANA_TUCKER_SP_TENSOR_HPP
#include "def.hpp"
#include "read_data.hpp"
#include "sp_operator.hpp"

template<typename Ty>
class SpTensor;

class index_buffer {
public:
    size_t* pos;
    size_t* index;
    index_buffer(size_t len, size_t nnz) {
        this->pos = new size_t[len + 1];
        this->index = new size_t[nnz];
    };
};

template<typename Ty>
class SpTensor {
private:
    size_t **index_lists_;
    Ty * vals_;
    shape_t dims_;
    size_t ndim_;
    size_t nnz_;

    Communicator<Ty> *comm_;
public:
    SpTensor();
    SpTensor(data_buffer<Ty> *);
    ~SpTensor();
    inline Sp_Operator<Ty> *op();
    [[nodiscard]] inline size_t ndim() const;
    [[nodiscard]] inline size_t nnz() const;
    [[nodiscard]] inline const shape_t &shape() const;
    [[nodiscard]] inline size_t** index_lists() const;
    inline const Ty* vals() const;
    inline Communicator<Ty> *comm() const;
    std::vector<index_buffer> v_index_buffer;
};

template<typename Ty>
SpTensor<Ty>::~SpTensor() {
    free(this->vals_);
    for (size_t i = 0; i < this->ndim_; i++){
        free(this->index_lists_[i]);
    }
    free(this->index_lists_);
}

template<typename Ty>
SpTensor<Ty>::SpTensor() {
    this->index_lists_ = nullptr;
    this->vals_ = nullptr;
    this->ndim_ = nullptr;
    this->nnz_ = nullptr;
    this->dims_ = shape_t();
}

template<typename Ty>
SpTensor<Ty>::SpTensor(data_buffer<Ty> *dbf) {
    this->nnz_ = dbf->nnz;
    this->ndim_ = dbf->ndim;
    this->index_lists_ = dbf->index_lists;
    this->vals_ = dbf->vals;
    this->dims_ = shape_t();
    std::vector<size_t*> v_start_index;

    for (size_t i = 0; i < dbf->ndim; i++) {
        this->dims_.push_back(dbf->dims[i]);
    }
    for (size_t i = 0; i < this->ndim(); i++) {
        index_buffer new_buffer(this->shape()[i], this->nnz());
        auto start_index = (size_t*) malloc(sizeof(size_t) * (this->shape()[i] + 1));
        for (size_t j = 0; j < this->shape()[i] + 1; j++) {
            new_buffer.pos[j] = 0;
        }
        v_start_index.push_back(start_index);
        this->v_index_buffer.push_back(new_buffer);
    }

    for (size_t i = 0; i < this->nnz(); i++) {
        for (size_t j = 0; j < this->ndim(); j++) {
            auto index = this->index_lists()[j][i];
            this->v_index_buffer[j].pos[index] += 1;
        }
    }

    for (size_t i = 0; i < this->ndim(); i++) {
        auto start_index = v_start_index[i];
        start_index[0] = 0;
        for (size_t j = 0; j < this->shape()[i]; j++) {
            start_index[j + 1] = start_index[j] + this->v_index_buffer[i].pos[j];
        }
        for (size_t j = 0; j < this->shape()[i] + 1; j++) {
            this->v_index_buffer[i].pos[j] = start_index[j];
        }
    }

    for (size_t i = 0; i < this->ndim(); i++) {
        auto start_index = v_start_index[i];
        auto index_list = this->index_lists()[i];
        for (size_t j = 0; j < this->nnz(); j++) {
            this->v_index_buffer[i].index[start_index[index_list[j]]++] = j;
        }
    }

    for (size_t i = 0; i < this->ndim(); i++) {
        free(v_start_index[i]);
    }
}

template<typename Ty>
inline size_t SpTensor<Ty>::ndim() const {
    return this->ndim_;
}

template<typename Ty>
inline size_t SpTensor<Ty>::nnz() const {
    return this->nnz_;
}

template<typename Ty>
inline const shape_t & SpTensor<Ty>::shape() const {
    return this->dims_;
}

template<typename Ty>
inline size_t ** SpTensor<Ty>::index_lists() const {
    return this->index_lists_;
}

template<typename Ty>
inline const Ty* SpTensor<Ty>::vals() const {
    return this->vals_;
}

template<typename Ty>
inline Communicator<Ty> *SpTensor<Ty>::comm() const {
    return this->comm_;
}

#include "sp_tensor_mpi.tpp"

#endif //DIANA_TUCKER_SP_TENSOR_HPP
