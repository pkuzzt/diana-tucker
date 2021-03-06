#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#include <omp.h>

namespace Function{
    template<typename Ty>
    double fnorm(const SpTensor<Ty> &A) {
        double ret = 0.0;
        for (size_t i = 0; i < A.nnz(); i++) {
            ret += A.vals()[i] * A.vals()[i];
        }
        A.comm()->allreduce_inplace(&ret, 1, MPI_SUM);
        return sqrt(ret);
    }

    template<typename Ty>
    Tensor<Ty>
    ttmc(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
         const std::vector<size_t> &idx, Distribution* distribution, bool to_permu) {
        Summary::start(METHOD_NAME);
        shape_t local_shape, local_start, local_end;
        shape_t global_shape = A.shape();
        int tag_index = 0, tag_data = 1;
        auto send_index = new MPI_Request;
        auto send_data = new MPI_Request;
        auto recv_index = new MPI_Request;
        auto recv_data = new MPI_Request;
        auto is_contract = new bool[A.ndim()];

        for (auto i : idx) {
            global_shape[i] = M[i].shape()[0];
        }

        for (size_t i = 0; i < A.ndim(); i++)
            is_contract[i] = false;
        if (!to_permu) {
            for (auto i : idx) {
                is_contract[i] = true;
            }
        }
        else {
            for (size_t i = 0; i < idx.size(); i++) {
                is_contract[i] = true;
            }
        }
        distribution->get_local_data(global_shape, local_shape, local_start, local_end);
        auto *index_buffer_recv = new size_t[3 * A.ndim() + 1];
        auto *index_buffer_send = new size_t[3 * A.ndim() + 1];

        auto current_start = index_buffer_recv;
        auto current_end = index_buffer_recv + A.ndim();
        auto current_stride = index_buffer_recv + 2 * A.ndim();
        auto current_size = index_buffer_recv + 3 * A.ndim();

        shape_t permu;
        if (to_permu) {
            for (auto i : idx) {
                permu.push_back(i);
            }
            for (size_t i = 0; i < A.ndim(); i++) {
                bool in_idx = false;
                for (auto j : idx) {
                    if (i == j) {
                        in_idx = true;
                        break;
                    }
                }
                if (!in_idx) {
                    permu.push_back(i);
                }
            }
        }
        else {
            for (size_t i = 0; i < A.ndim(); i++) {
                permu.push_back(i);
            }
        }

        size_t local_size = 1, max_size;
        for (auto mode : local_shape)
            local_size *= mode;

        for (size_t i = 0; i < A.ndim(); i++) {
            index_buffer_send[i] = local_start[permu[i]];
            index_buffer_send[i + A.ndim()] = local_end[permu[i]];
        }

        index_buffer_send[2 * A.ndim()] = 1;
        for (size_t i = 1; i < A.ndim(); i++) {
            index_buffer_send[i + 2 * A.ndim()] = local_shape[permu[i - 1]] * index_buffer_send[i - 1 + 2 * A.ndim()];
        }

        index_buffer_send[3 * A.ndim()] = local_size;

        Communicator<size_t>::allreduce(&local_size, &max_size, 1, MPI_MAX);

        auto data_buffer_send = new Ty[max_size];
        auto data_buffer_recv = new Ty[max_size];
        auto data_buffer_tmp = new Ty[max_size];
        int rank = mpi_rank(), size = mpi_size();
        int prev_rank = rank > 0 ? rank - 1 : size - 1;
        int next_rank = rank < size - 1 ? rank + 1 : 0;


        Communicator<size_t>::isend(send_index, index_buffer_send, 3 * (int) A.ndim() + 1, next_rank, MPI_COMM_WORLD, tag_index);


        for (int cycle = 0; cycle < size; cycle++){
            Communicator<size_t>::irecv(recv_index, index_buffer_recv, 3 * (int) A.ndim() + 1, prev_rank, MPI_COMM_WORLD, tag_index);
            Communicator<size_t>::wait(recv_index);
            if (cycle > 0)
                Communicator<Ty>::irecv(recv_data, data_buffer_recv, (int) *current_size, prev_rank, MPI_COMM_WORLD, tag_data);

            for (size_t i = 0; i < *current_size; i++)
                data_buffer_tmp[i] = 0;

            if (!to_permu) {
                for (size_t i = 0; i < A.nnz(); i++) {
                    auto val = A.vals()[i];
                    shape_t index;
                    for (size_t j = 0; j < A.ndim(); j++) {
                        index.push_back(A.index_lists()[j][i]);
                    }
                    Function::add_outer_product<Ty>(data_buffer_tmp, current_start, current_end, current_stride,
                                                    M, val, index, A.ndim() - 1, is_contract);
                }
            }
            else {
                for (size_t i = 0; i < A.nnz(); i++) {
                    auto val = A.vals()[i];
                    shape_t index;
                    for (size_t j = 0; j < A.ndim(); j++) {
                        index.push_back(A.index_lists()[permu[j]][i]);
                    }
                    Function::add_outer_product_permu<Ty>(data_buffer_tmp, current_start, current_end, current_stride,
                                                    M, val, index, A.ndim() - 1, is_contract, permu);
                }
            }

            if (cycle > 0) {
                Communicator<Ty>::wait(recv_data);
                Communicator<Ty>::wait(send_data);
                for (size_t i = 0; i < *current_size; i++) {
                    data_buffer_send[i] = data_buffer_recv[i] + data_buffer_tmp[i];
                }
            }
            else {
                for (size_t i = 0; i < *current_size; i++) {
                    data_buffer_send[i] = data_buffer_tmp[i];
                }
            }

            if (cycle < size - 1) {
                for (size_t i = 0; i < 3 * A.ndim() + 1; i++)
                    index_buffer_send[i] = index_buffer_recv[i];

                Communicator<size_t>::isend(send_index, index_buffer_send,  3 * (int) A.ndim() + 1, next_rank, MPI_COMM_WORLD, tag_index);
                Communicator<Ty>::isend(send_data, data_buffer_send, (int) *current_size, next_rank, MPI_COMM_WORLD, tag_data);
            }
        }

        delete [] data_buffer_recv;
        Tensor<Ty> ret(distribution, global_shape, false);

        if (to_permu) {
            shape_t ipermu;
            shape_t permu_shape;
            for (size_t i = 0; i < A.ndim(); i++) {
                ipermu.push_back(0);
                permu_shape.push_back(local_shape[permu[i]]);
            }
            for (size_t i = 0; i < A.ndim(); i++)
                ipermu[permu[i]] = i;

            Function::permutate<Ty>(data_buffer_send, ret.data(), permu_shape, ipermu);
        }
        else {
            for (size_t i = 0; i < *current_size; i++)
                ret.data()[i] = data_buffer_send[i];
        }

        delete [] index_buffer_recv;
        delete [] index_buffer_send;
        delete [] data_buffer_send;
        delete [] data_buffer_tmp;
        delete send_index;
        delete send_data;
        delete recv_index;
        delete recv_data;
        Summary::end(METHOD_NAME);
        return ret;
    }

    template<typename Ty>
    Tensor<Ty>
    mettmc(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
               const std::vector<size_t> &idx, Distribution* distribution, bool to_permu, size_t level) {
        Summary::start(METHOD_NAME);
        shape_t tmp = idx;
        shape_t new_idx;
        std::sort(tmp.begin(), tmp.end(),  [&A, &M](size_t x, size_t y){return A.shape()[x] / M[x].shape()[0] < A.shape()[y] / M[y].shape()[0];});
        for (size_t i = level; i < tmp.size(); i++) {
            new_idx.push_back(tmp[i]);
        }
        auto Y = Function::ttmc(A, M, new_idx, distribution, to_permu);
        for (size_t i = 0; i < level; i++) {
            Y = Function::ttm<Ty>(Y, M[tmp[i]], tmp[i]);
        }
        Summary::end(METHOD_NAME);
        return Y;
    }

    template<typename Ty>
    void permutate(Ty* data1, Ty* data2, shape_t & shape, shape_t & permu) {
        Summary::start(METHOD_NAME);
        shape_t stride;
        size_t size;
        size_t tmp = 1;
        for (unsigned long i : shape) {
            stride.push_back(tmp);
            tmp *= i;
        }
        size = tmp;
        for (size_t i = 0; i < size; i++) {
            tmp = i;
            size_t index = 0;
            for (size_t j = 0; j < shape.size(); j++) {
                index += stride[permu[j]] * (tmp % shape[permu[j]]);
                tmp /= shape[permu[j]];
            }
            data2[i] = data1[index];
        }
        Summary::end(METHOD_NAME);
    }

    template<typename Ty>
    Tensor<Ty>
    ttmc_mTR(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M, size_t n, const Tensor<Ty> &R) {
        Summary::start(METHOD_NAME);
        assert(A.shape()[n] == R.shape()[1]);
        size_t len1 = 1, len2 = R.shape()[0];
        shape_t permu, core_shape, core_stride;
        for (size_t i = 0; i < A.ndim(); i++) {
            if (i != n) {
                core_shape.push_back(M[i].shape()[0]);
                core_stride.push_back(len1);
                len1 *= M[i].shape()[0];
                permu.push_back(i);
            }
        }
        permu.push_back(n);

        auto global_distri = new DistributionGlobal();
        Tensor<Ty> ret(global_distri, {len1, len2}, true);

        auto tmp = (Ty*) malloc(sizeof(Ty) * len1);

        shape_t index;
        for (size_t i = 0; i < A.ndim() - 1; i++) {
            index.push_back(0);
        }

        for (size_t i = 0; i < len1; i++) {
            tmp[i] = 0;
        }

        for (size_t i = 0; i < A.nnz(); i++) {
            for (size_t dim = 0; dim < A.ndim(); dim++) {
                index[dim] = A.index_lists()[permu[dim]][i];
            }
            auto val = A.vals()[i];
            Function::add_outer_product_all(tmp ,core_shape, core_stride, M, val, index, A.ndim() - 2, permu);
            if (i < A.nnz() - 1 && A.index_lists()[n][i + 1] != A.index_lists()[n][i]) {
                for (size_t j = 0; j < len1; j++) {
                    for (size_t k = 0; k < len2; k++) {
                        ret.data()[k * len1 + j] += tmp[j] * R.data()[i * len2 + k];
                    }
                }
            }
        }

        free(tmp);
        Communicator<Ty>::allreduce_inplace(ret.data(), (int) ret.size(), MPI_SUM, MPI_COMM_WORLD);
        Summary::end(METHOD_NAME);
        return ret;
    }

    template<typename Ty>
    void ttmc_mTL(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M, size_t n, const Tensor<Ty> &L, Tensor<Ty> &ret) {
        Summary::start(METHOD_NAME);
        assert(L.shape()[1] == M[n].shape()[0]);
        auto len1 = L.shape()[1];
        size_t tmp_len = 1;

        shape_t permu, core_shape, core_stride;
        for (size_t i = 0; i < A.ndim(); i++) {
            if (i != n) {
                core_shape.push_back(M[i].shape()[0]);
                core_stride.push_back(len1);
                tmp_len *= M[i].shape()[0];
                permu.push_back(i);
            }
        }
        assert(L.shape()[0] == tmp_len);

        #pragma omp parallel for
        for (size_t i = 0; i < ret.size(); i++) {
            ret.data()[i] = 0;
        }

        size_t nstage = 10;
        nstage = std::min(nstage, DIANA_CEILDIV(A.shape()[n], 10000));
        auto request_list = new MPI_Request[nstage];
        #pragma omp parallel
        {
            shape_t index;
            for (size_t i = 0; i < A.ndim() - 1; i++) {
                index.push_back(0);
            }

            auto tmp = (Ty*) malloc(sizeof(Ty) * tmp_len);

            for (size_t stage = 0; stage < nstage; stage++) {
                size_t start = stage * A.shape()[n] / nstage;
                size_t end = (stage + 1) * A.shape()[n] / nstage;

                #pragma omp for schedule(dynamic)
                for (size_t i = start; i < end; i++) {
                    if (A.v_index_buffer[n].pos[i + 1] == A.v_index_buffer[n].pos[i])
                        continue;

                    for (size_t j = 0; j < tmp_len; j++) {
                        tmp[j] = 0;
                    }

                    for (size_t j = A.v_index_buffer[n].pos[i]; j < A.v_index_buffer[n].pos[i + 1]; j++){
                        auto k = A.v_index_buffer[n].index[j];
                        for (size_t dim = 0; dim < A.ndim() - 1; dim++) {
                            index[dim] = A.index_lists()[permu[dim]][k];
                        }
                        auto val = A.vals()[k];
                        Function::add_outer_product_all(tmp, core_shape, core_stride, M, val, index, A.ndim() - 2, permu);
                    }

                    auto data = ret.data() + len1 * i;
                    for (size_t j = 0; j < len1; j++) {
                        auto data1 = L.data() + tmp_len * j;
                        for (size_t k = 0; k < tmp_len; k++) {
                            data[j] += tmp[k] * data1[k];
                        }
                    }
                }
                #pragma omp master
                {
                    Communicator<Ty>::iallreduce_inplace(ret.data() + start * len1, (int) len1 * (int) (end - start), MPI_SUM, MPI_COMM_WORLD, request_list + stage);
                }
            }

            free(tmp);
        }
        //Communicator<Ty>::allreduce_inplace(ret.data(), (int) ret.size(), MPI_SUM, MPI_COMM_WORLD);
        Communicator<Ty>::waitall((int) nstage, request_list);
        delete [] request_list;
        Summary::end(METHOD_NAME);
    }

    template<typename Ty>
    void add_outer_product_all(Ty *data, const shape_t &shape, const shape_t &stride, const std::vector<Tensor<Ty>> &M, Ty val,
                               const shape_t &index, size_t dim, shape_t &permu){
        if (dim == 0) {
            auto vec0 = M[permu[0]].data() + index[0] * M[permu[0]].shape()[0];
            auto size0 = M[permu[dim]].shape()[0];
            for (size_t i = 0; i < size0; i++) {
                data[i] += val * vec0[i];
            }
        }
        else if (dim == 1) {
            auto vec1 = M[permu[1]].data() + index[1] * M[permu[1]].shape()[0];
            auto vec0 = M[permu[0]].data() + index[0] * M[permu[0]].shape()[0];
            auto size1 = shape[1];
            auto size0 = shape[0];
            for (size_t j = 0; j < size1; j++) {
                auto val1 = val * vec1[j];
                auto data1 = data + stride[1] * j;
                for (size_t i = 0; i < size0; i++) {
                    data1[i] += val1 * vec0[i];
                }
            }
        }
        else if (dim == 2) {
            auto vec2 = M[permu[2]].data() + index[2] * M[permu[2]].shape()[0];
            auto vec1 = M[permu[1]].data() + index[1] * M[permu[1]].shape()[0];
            auto vec0 = M[permu[0]].data() + index[0] * M[permu[0]].shape()[0];
            auto size2 = shape[2];
            auto size1 = shape[1];
            auto size0 = shape[0];
            for (size_t k = 0; k < size2; k++) {
                auto val2 = val * vec2[k];
                auto data2 = data + stride[2] * k;
                for (size_t j = 0; j < size1; j++) {
                    auto val1 = val2 * vec1[j];
                    auto data1 = data2 + stride[1] * j;
                    for (size_t i = 0; i < size0; i++) {
                        data1[i] += val1 * vec0[i];
                    }
                }
            }
        }
        else {
            auto size_dim = shape[dim];
            auto J_dim = M[permu[dim]].shape()[0];
            for (size_t i = 0; i < size_dim; i++) {
                add_outer_product_all<Ty>(data + stride[dim] * i, shape, stride, M, val * M[permu[dim]].data()[index[dim] * J_dim + i],
                                            index, dim - 1, permu);
            }
        }
    }

    template<typename Ty>
    void add_outer_product(Ty* data, const size_t *start_index, const size_t *end_index, const size_t *stride,
                           const std::vector<Tensor<Ty>> &M, Ty val, const shape_t &index, size_t dim, const bool * is_contract) {
        auto size_dim = end_index[dim] - start_index[dim];
        auto J_dim = M[dim].shape()[0];

        if (dim == 0) {
            if (is_contract[dim]) {
                for (size_t i = 0; i < size_dim; i++) {
                    data[i] += val * M[dim].data()[index[dim] * J_dim + start_index[dim] + i];
                }
            }
            else {
                if (index[dim] < start_index[dim] || index[dim] >= end_index[dim])
                    return;
                else
                    data[index[dim] - start_index[dim]] += val;
            }
        }
        else {
            if (is_contract[dim]) {
                for (size_t i = 0; i < size_dim; i++) {
                    add_outer_product<Ty>(data + stride[dim] * i, start_index, end_index, stride, M, val * M[dim].data()[index[dim] * J_dim + start_index[dim] + i],
                                      index, dim - 1, is_contract);
                }
            }
            else {
                if (index[dim] < start_index[dim] || index[dim] >= end_index[dim]) {
                    return;
                }
                add_outer_product<Ty>(data + stride[dim] * (index[dim] - start_index[dim]), start_index, end_index, stride, M, val,
                                  index, dim - 1, is_contract);
            }
        }
    }

    template<typename Ty>
    void add_outer_product_permu(Ty* data, const size_t *start_index, const size_t *end_index, const size_t *stride,
                                 const std::vector<Tensor<Ty>> &M, Ty val, const shape_t &index, size_t dim, const bool * is_contract, shape_t & permu) {
        auto size_dim = end_index[dim] - start_index[dim];
        auto J_dim = M[permu[dim]].shape()[0];

        if (dim == 0) {
            if (is_contract[dim]) {
                Ty * tmp_data = M[permu[dim]].data() + index[dim] * J_dim + start_index[dim];
                for (size_t i = 0; i < size_dim; i++) {
                    data[i] += val * tmp_data[i];
                }
                // TODO: Add SIMD
            }
            else {
                if (index[dim] < start_index[dim] || index[dim] >= end_index[dim])
                    return;
                else
                    data[index[dim] - start_index[dim]] += val;
            }
        }
        else {
            if (is_contract[dim]) {
                for (size_t i = 0; i < size_dim; i++) {
                    add_outer_product_permu<Ty>(data + stride[dim] * i, start_index, end_index, stride, M, val * M[permu[dim]].data()[index[dim] * J_dim + start_index[dim] + i],
                                          index, dim - 1, is_contract, permu);
                }
            }
            else {
                if (index[dim] < start_index[dim] || index[dim] >= end_index[dim]) {
                    return;
                }
                add_outer_product_permu<Ty>(data + stride[dim] * (index[dim] - start_index[dim]), start_index, end_index, stride, M, val,
                                      index, dim - 1, is_contract, permu);
            }
        }
    }
}

#pragma clang diagnostic pop