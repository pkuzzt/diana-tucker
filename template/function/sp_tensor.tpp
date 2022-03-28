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
    ttmNTc(const SpTensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
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
            global_shape[i] = M[i].shape()[1];
        }

        for (size_t i = 0; i < A.ndim(); i++)
            is_contract[i] = false;
        if (!to_permu) {
            for (auto i : idx) {
                global_shape[i] = M[i].shape()[1];
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
                Communicator<Ty>::barrier(MPI_COMM_WORLD);
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
            for (size_t i = 0; i < A.ndim(); i++)
                ipermu.push_back(0);
            for (size_t i = 0; i < A.ndim(); i++)
                ipermu[permu[i]] = i;
            Function::permutate<Ty>(data_buffer_send, ret.data(), local_shape, ipermu);
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
    void permutate(Ty* data1, Ty* data2, shape_t & shape, shape_t & permu) {
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
    }
    template<typename Ty>
    void add_outer_product(Ty* data, const size_t *start_index, const size_t *end_index, const size_t *stride,
                           const std::vector<Tensor<Ty>> &M, Ty val, const shape_t &index, size_t dim, const bool * is_contract) {
        auto size_dim = end_index[dim] - start_index[dim];
        auto I_dim = M[dim].shape()[0];

        if (dim == 0) {
            if (is_contract[dim]) {
                for (size_t i = 0; i < size_dim; i++) {
                    data[i] += val * M[dim].data()[index[dim] + I_dim * (start_index[dim] + i)];
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
                    add_outer_product<Ty>(data + stride[dim] * i, start_index, end_index, stride, M, val * M[dim].data()[index[dim] + I_dim * (start_index[dim] + i)],
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
        auto I_dim = M[permu[dim]].shape()[0];

        if (dim == 0) {
            if (is_contract[dim]) {
                for (size_t i = 0; i < size_dim; i++) {
                    data[i] += val * M[permu[dim]].data()[index[dim] + I_dim * (start_index[dim] + i)];
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
                    add_outer_product_permu<Ty>(data + stride[dim] * i, start_index, end_index, stride, M, val * M[permu[dim]].data()[index[dim] + I_dim * (start_index[dim] + i)],
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
