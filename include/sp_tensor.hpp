//
// Created by 30250 on 2022/3/17.
//

#ifndef DIANA_TUCKER_SP_TENSOR_HPP
#define DIANA_TUCKER_SP_TENSOR_HPP
#include "def.hpp"
#include "read_data.hpp"
#include "sp_operator.hpp"
#include <algorithm>
#include <string.h>

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

public:
    size_t slice_mode;
    size_t slice_start;
    size_t slice_end;
    shape_t send_to_list;
    shape_t recv_from_list;
    std::vector<size_t*> v_index_list;
    SpTensor();
    SpTensor(data_buffer<Ty> *);
    ~SpTensor();
    inline Sp_Operator<Ty> *op();
    [[nodiscard]] inline size_t ndim() const;
    [[nodiscard]] inline size_t nnz() const;
    [[nodiscard]] inline const shape_t &shape() const;
    [[nodiscard]] inline size_t** index_lists() const;
    inline const Ty* vals() const;
};

template<typename Ty>
SpTensor<Ty>::~SpTensor() {
    free(this->vals_);
    for (size_t i = 0; i < this->ndim_; i++){
        free(this->index_lists_[i]);
    }
    free(this->index_lists_);
    for (size_t i = 0; i < this->v_index_list.size(); i++) {
        free(this->v_index_list[i]);
    }
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
    Summary::start(METHOD_NAME);
    auto rank = (size_t) mpi_rank();
    auto size = (size_t) mpi_size();

    // bcast data
    Communicator<size_t>::bcast(&dbf->nnz, 1, 0, MPI_COMM_WORLD);
    Communicator<size_t>::bcast(&dbf->ndim, 1, 0, MPI_COMM_WORLD);
    this->ndim_ = dbf->ndim;

    if (rank != 0) {
        for (size_t i = 0; i < this->ndim(); i++) {
            this->dims_.push_back(0);
        }
    }
    else {
        for (size_t i = 0; i < this->ndim(); i++) {
            this->dims_.push_back(dbf->dims[i]);
        }
    }
    Communicator<size_t>::bcast(this->dims_.data(), (int) this->ndim(), 0, MPI_COMM_WORLD);

    // set local nnz
    size_t start_index = rank * dbf->nnz / size;
    size_t end_index = (rank + 1) * dbf->nnz / size;
    this->nnz_ = end_index - start_index;

    // malloc
    this->vals_ = (Ty*) malloc(sizeof(Ty) * this->nnz());
    this->index_lists_ = (size_t**) malloc(sizeof(size_t*) * this->ndim());
    for (size_t i = 0; i < this->ndim(); i++) {
        this->index_lists_[i] = (size_t*) malloc(sizeof(size_t) * this->nnz());
    }

    if (rank == 0) {
        // get slice mode
        size_t slice_mode = 0;
        size_t slice_size = 0;
        for (size_t i = 0; i < this->ndim(); i++) {
            if (slice_size < this->shape()[i]) {
                slice_mode = i;
                slice_size = this->shape()[i];
            }
        }

        auto slice_list = (size_t*) malloc(sizeof(size_t) * slice_size);
        auto start_index1 = (size_t*) malloc(sizeof(size_t) * (slice_size + 1));
        auto tmp = (size_t*) malloc(sizeof(size_t) * (slice_size + 1));
        auto reordered_vals = (Ty*) malloc(sizeof(Ty) * dbf->nnz);

        for (size_t i = 0; i < this->shape()[slice_mode]; i++) {
            slice_list[i] = 0;
        }
        for (size_t i = 0; i < dbf->nnz; i++) {
            auto index = dbf->index_lists[slice_mode][i];
            slice_list[index]++;
        }

        start_index1[0] = 0;
        for (size_t i = 0; i < slice_size; i++) {
            start_index1[i + 1] = start_index1[i] + slice_list[i];
        }

        // reorder
        for (size_t i = 0; i < slice_size + 1; i++) {
            tmp[i] = start_index1[i];
        }

        for (size_t i = 0; i < dbf->nnz; i++) {
            auto slice_mode_index = dbf->index_lists[slice_mode][i];
            auto index = tmp[slice_mode_index]++;
            reordered_vals[index] = dbf->vals[i];
        }
        for (size_t i = 0; i < this->nnz(); i++) {
            this->vals_[i] = reordered_vals[i];
        }

        auto requests = new MPI_Request[size - 1];

        // send values
        for (size_t i = 1; i < size; i++) {
            auto start = i * dbf->nnz / size;
            auto end = (i + 1) * dbf->nnz / size;
            Communicator<Ty>::isend(requests + i - 1, reordered_vals + start, (int) (end - start), i, MPI_COMM_WORLD, 0);
        }
        Communicator<Ty>::waitall((int) size - 1, requests);
        free(reordered_vals);

        auto reordered_indices = (size_t*) malloc(sizeof(size_t) * dbf->nnz);
        for (size_t i = 0; i < this->ndim(); i++) {
            // reorder
            for (size_t j = 0; j < slice_size + 1; j++) {
                tmp[j] = start_index1[j];
            }
            for (size_t j = 0; j < dbf->nnz; j++) {
                auto slice_mode_index = dbf->index_lists[slice_mode][j];
                auto index = tmp[slice_mode_index]++;
                reordered_indices[index] = dbf->index_lists[i][j];
            }

            // send indices
            for (size_t j = 0; j < this->nnz(); j++) {
                this->index_lists_[i][j] = reordered_indices[j];
            }

            for (size_t j = 1; j < size; j++) {
                auto start = j * dbf->nnz / size;
                auto end = (j + 1) * dbf->nnz / size;
                Communicator<size_t>::isend(requests + j - 1, reordered_indices + start, (int) (end - start), j, MPI_COMM_WORLD, i + 1);
            }
            Communicator<size_t>::waitall((int) size - 1, requests);
        }

        free(reordered_indices);
        free(slice_list);
        free(start_index1);
        free(tmp);
        free(dbf->vals);
        for (size_t i = 0; i < this->ndim(); i++) {
            free(dbf->index_lists[i]);
        }
        free(dbf->index_lists);
        delete [] requests;

        this->slice_mode = slice_mode;
    }
    else {
        auto requests = new MPI_Request[this->ndim() + 1];
        // receive values
        Communicator<Ty>::recv(this->vals_, (int) this->nnz(), 0, MPI_COMM_WORLD, 0);
        for (size_t i = 0; i < this->ndim(); i++) {
            Communicator<size_t>::recv(this->index_lists_[i], (int) this->nnz(), 0, MPI_COMM_WORLD, i + 1);
        }
        delete [] requests;
    }

    // Split Communicator
    Communicator<size_t>::bcast(&this->slice_mode, 1, 0, MPI_COMM_WORLD);
    this->slice_start = this->index_lists()[this->slice_mode][0];
    this->slice_end = this->index_lists()[this->slice_mode][this->nnz() - 1];

    auto slice_start_list = (size_t*) malloc(sizeof(size_t) * size);
    auto slice_end_list = (size_t*) malloc(sizeof(size_t) * size);

    Communicator<size_t>::allgather(&this->slice_start, 1, slice_start_list, MPI_COMM_WORLD);
    Communicator<size_t>::allgather(&this->slice_end, 1, slice_end_list, MPI_COMM_WORLD);


    for (size_t i = 0; i < size - 1; i++) {
        if (slice_end_list[i] == slice_start_list[i + 1]) {
            auto start = i;
            auto end = i + 1;
            while (slice_start_list[end + 1] == slice_end_list[start]) {
                end++;
            }
            if (rank == start) {
                for (size_t j = start + 1; j <= end; j++)
                    this->recv_from_list.push_back(j);
            }
            else if (rank > start && rank <= end){
                this->send_to_list.push_back(start);
            }
            i = end - 1;
        }
    }
    free(slice_start_list);
    free(slice_end_list);


    // set index vector
    auto start_index_list = new size_t[this->shape()[this->slice_mode] + 1];

    for (size_t i = 0; i < this->ndim(); i++) {
        auto index_ptr = (size_t*) malloc(sizeof(size_t) * this->nnz());
        for (size_t j = 0; j <= this->shape()[i]; j++) {
            start_index_list[j] = 0;
        }
        for (size_t j = 0; j < this->nnz(); j++) {
            auto idx_i = this->index_lists()[i][j];
            start_index_list[idx_i + 1]++;
        }
        for (size_t j = 1; j <= this->shape()[i]; j++) {
            start_index_list[j] += start_index_list[j - 1];
        }
        for (size_t j = 0; j < this->nnz(); j++) {
            auto idx_i = this->index_lists()[i][j];
            index_ptr[start_index_list[idx_i]++] = j;
        }
        this->v_index_list.push_back(index_ptr);
    }

    delete [] start_index_list;
    Summary::end(METHOD_NAME);
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

#include "sp_tensor_mpi.tpp"

#endif //DIANA_TUCKER_SP_TENSOR_HPP
