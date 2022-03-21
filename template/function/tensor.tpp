#include "tensor.hpp"
#include "logger.hpp"
#include "summary.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>

namespace Function {
    /**
     * @brief Calculate \f$ \bm{\mathcal{A}}_{(n)} \bm{\mathcal{A}}_{(n)}^T \f$,
     * where \f$ \bm{\mathcal{A}} \f$ is a tensor.
     * @tparam Ty
     * @param A
     * @param n
     * @return
     */
    template<typename Ty>
    Tensor<Ty> gram(const Tensor<Ty> &A, size_t n) {
        if (A.distribution()->type() == Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            // Initialization.
            auto *distrib = (DistributionCartesianBlock *) A.distribution();
            shape_t par = distrib->partition();
            shape_t coord = distrib->coordinate();
            const size_t kParN = par[n];
            const size_t kGlobalShapeN = A.shape_global()[n];
            const size_t kLocalShapeN = A.shape()[n];
            // Split communicator.
            auto[new_color, new_rank] = distrib->process_fiber(n);
            MPI_Comm comm_fiber = distrib->process_fiber_comm(n);
            // Allocate double buffer.
            size_t max_size = A.size();
            Ty *data_A = A.data();
            Ty *databuf[2]; // Double buffer
            Communicator<size_t>::allreduce_inplace(&max_size, 1,
                                                    MPI_MAX, comm_fiber);
            databuf[0] = A.op()->alloc(max_size);
            databuf[1] = A.op()->alloc(max_size);
            // Allocate gram_buffer.
            const size_t row_length = kLocalShapeN;
            const size_t col_length = A.size() / kLocalShapeN;
            size_t *all_row_length = Operator<size_t>::alloc(kParN);
            size_t *gram_buf_start = Operator<size_t>::alloc(kParN);
            size_t gram_buf_size;
            auto gram_buf_point = (size_t) new_rank;
            Communicator<size_t>::allgather(&row_length, 1, all_row_length,
                                            comm_fiber);
            gram_buf_start[0] = 0;
            for (size_t i = 1; i < kParN; i++) {
                gram_buf_start[i] =
                        gram_buf_start[i - 1] + all_row_length[i - 1];
            }
            gram_buf_size = row_length * kGlobalShapeN;
            Ty *gram_buf = A.op()->alloc(gram_buf_size);
            // Allocate A_buf.
            Ty *A_buf = A.op()->alloc(A.size());
            // Initialize data transpose.
            const int send_to_proc_id =
                    ((int) new_rank - 1 + (int) kParN) % (int) kParN;
            const int recv_from_proc_id = ((int) new_rank + 1) % (int) kParN;
            // Matricization
            A.op()->tenmat(A_buf, data_A, A.shape(), n);
            A.op()->mcpy(databuf[0], A_buf, A.size());
            // Do gram
            MPI_Request *request_send = A.comm()->new_request();
            MPI_Request *request_recv = A.comm()->new_request();
            for (size_t i = 0; i < kParN; i++) {
                if (i != 0) {
                    A.comm()->wait(request_send);
                    A.comm()->wait(request_recv);
                }
                if (i != kParN - 1) {
                    A.comm()->isend(request_send, databuf[i % 2],
                                    (int) max_size,
                                    send_to_proc_id, comm_fiber);
                    A.comm()->irecv(request_recv,
                                    databuf[(i + 1) % 2],
                                    (int) max_size,
                                    recv_from_proc_id, comm_fiber);
                }
                A.op()->matmulNT(gram_buf +
                                 gram_buf_start[gram_buf_point] * kLocalShapeN,
                                 A_buf, databuf[i % 2], row_length,
                                 all_row_length[gram_buf_point], col_length);
                gram_buf_point = (gram_buf_point + 1) % kParN;
            }
            // Allreduce.
            MPI_Comm comm_line = A.comm()->comm_split(new_rank, new_color);
            A.comm()->allreduce_inplace(gram_buf, (int) gram_buf_size, MPI_SUM,
                                        comm_line);
            // Gather.
            Tensor<Ty> gram({kGlobalShapeN, kGlobalShapeN}, false);
            Ty *gram_data = gram.data();
            int *recvcount = Operator<int>::alloc(kParN);
            int *displs = Operator<int>::alloc(kParN);
            for (size_t i = 0; i < kParN; i++) {
                recvcount[i] = (int) all_row_length[i];
                displs[i] = (int) gram_buf_start[i];
            }
            for (size_t i = 0; i < kGlobalShapeN; i++) {
                A.comm()->allgatherv(gram_buf + i * kLocalShapeN,
                                     (int) kLocalShapeN,
                                     gram_data + i * kGlobalShapeN,
                                     recvcount, displs, comm_fiber);
            }
            // Free buffers.
            A.op()->free(databuf[0]);
            A.op()->free(databuf[1]);
            Operator<size_t>::free(all_row_length);
            Operator<size_t>::free(gram_buf_start);
            A.op()->free(gram_buf);
            A.op()->free(A_buf);
            A.comm()->free_request(request_send);
            A.comm()->free_request(request_recv);
            Summary::end(METHOD_NAME);
            return gram;
        }
        error("Invalid input or not implemented yet.");
    }


    /**
     * @brief Calculate \f$ \bm{\mathcal{A}}_{(n)} \bm{\mathcal{B}}_{(n)}^T \f$,
     * where \f$ \bm{\mathcal{A}} \f$ and \f$ \bm{\mathcal{B}} \f$ are tensors.
     * @tparam Ty
     * @param A
     * @param B
     * @param n
     * @return
     */
    template<typename Ty>
    Tensor<Ty> ttt_except(const Tensor<Ty> &A, const Tensor<Ty> &B, size_t n) {
        if (A.distribution()->type() == Distribution::Type::kCartesianBlock
            ||
            B.distribution()->type() == Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            for (size_t i = 0; i < A.ndim(); i++) {
                if (i == n) continue;
                assert(A.shape_global()[i] == B.shape_global()[i]);
                assert(A.shape()[i] == B.shape()[i]);
            }
            // TODO: swap(A, B)
            // Initialization.
            auto *distrib = (DistributionCartesianBlock *) A.distribution();
            shape_t par = distrib->partition();
            shape_t coord = distrib->coordinate();
            const size_t kParN = par[n];
            const size_t kAGlobalShapeN = A.shape_global()[n];
            const size_t kALocalShapeN = A.shape()[n];
            const size_t kBGlobalShapeN = B.shape_global()[n];
            const size_t kBLocalShapeN = B.shape()[n];
            // Split communicator.
            auto[new_color, new_rank] = distrib->process_fiber(n);
            MPI_Comm comm_fiber = distrib->process_fiber_comm(n);
            // Allocate double buffer.
            size_t max_size = B.size();
            Ty *data_A = A.data();
            Ty *data_B = B.data();
            Ty *databuf[2]; // Double buffer
            Communicator<size_t>::allreduce_inplace(&max_size, 1,
                                                    MPI_MAX,
                                                    comm_fiber); // TODO: Optimize
            databuf[0] = B.op()->alloc(max_size);
            databuf[1] = B.op()->alloc(max_size);
            // Allocate gram_buffer.
            const size_t A_row_length = kALocalShapeN;
            const size_t B_row_length = kBLocalShapeN;
            const size_t col_length = A.size() / kALocalShapeN;
            size_t *all_B_row_length = Operator<size_t>::alloc(kParN);
            size_t *all_A_row_length = Operator<size_t>::alloc(kParN);
            size_t *gram_buf_start = Operator<size_t>::alloc(kParN);
            size_t gram_buf_size;
            auto gram_buf_point = (size_t) new_rank;
            Communicator<size_t>::allgather(&B_row_length, 1, all_B_row_length,
                                            comm_fiber); // TODO: Optimize
            Communicator<size_t>::allgather(&A_row_length, 1, all_A_row_length,
                                            comm_fiber); // TODO: Optimize
            gram_buf_start[0] = 0;
            for (size_t i = 1; i < kParN; i++) {
                gram_buf_start[i] =
                        gram_buf_start[i - 1] + all_B_row_length[i - 1];
            }
            gram_buf_size = kALocalShapeN * kBGlobalShapeN;
            Ty *gram_buf = A.op()->alloc(gram_buf_size);
            // Allocate A_buf.
            Ty *A_buf = A.op()->alloc(A.size());
            // Matricization
            // TODO: Local gram kernel.
            A.op()->tenmat(A_buf, data_A, A.shape(), n);
            B.op()->tenmat(databuf[0], data_B, B.shape(), n);
            // Initialize data transpose.
            // TODO: check MPI.
            const int send_to_proc_id =
                    ((int) new_rank - 1 + (int) kParN) % (int) kParN;
            const int recv_from_proc_id = ((int) new_rank + 1) % (int) kParN;
            // Do gram
            MPI_Request *request_send = A.comm()->new_request();
            MPI_Request *request_recv = A.comm()->new_request();
            for (size_t i = 0; i < kParN; i++) {
                if (i != 0) {
                    A.comm()->wait(request_send);
                    A.comm()->wait(request_recv);
                }
                if (i != kParN - 1) {
                    //TODO: check i.
                    A.comm()->isend(request_send, databuf[i % 2],
                                    (int) max_size,
                                    send_to_proc_id, comm_fiber);
                    A.comm()->irecv(request_recv,
                                    databuf[(i + 1) % 2],
                                    (int) max_size,
                                    recv_from_proc_id, comm_fiber);
                }
                A.op()->matmulNT(gram_buf +
                                 gram_buf_start[gram_buf_point] * kALocalShapeN,
                                 A_buf, databuf[i % 2], A_row_length,
                                 all_B_row_length[gram_buf_point], col_length);
                gram_buf_point = (gram_buf_point + 1) % kParN;
            }
            // Allreduce.
            // TODO: optimize.
            MPI_Comm comm_line = A.comm()->comm_split(new_rank, new_color);
            // TODO: Not inplace?
            A.comm()->allreduce_inplace(gram_buf, (int) gram_buf_size, MPI_SUM,
                                        comm_line);
            // Gather.
            // TODO: pack !!!!
            Tensor<Ty> gram({kAGlobalShapeN, kBGlobalShapeN}, false);
            Ty *gram_data = gram.data();
            int *recvcount = Operator<int>::alloc(kParN);
            int *displs = Operator<int>::alloc(kParN);
            recvcount[0] = (int) all_A_row_length[0];
            displs[0] = 0;
            for (size_t i = 1; i < kParN; i++) {
                recvcount[i] = (int) all_A_row_length[i];
                displs[i] = (int) displs[i - 1] + (int) all_A_row_length[i - 1];
            }
            for (size_t i = 0; i < kBGlobalShapeN; i++) {
                A.comm()->allgatherv(gram_buf + i * kALocalShapeN,
                                     (int) kALocalShapeN,
                                     gram_data + i * kAGlobalShapeN,
                                     recvcount, displs, comm_fiber);
            }
            // TODO: swap(A, B)
            // Free buffers.
            A.op()->free(databuf[0]);
            A.op()->free(databuf[1]);
            Operator<size_t>::free(all_B_row_length);
            Operator<size_t>::free(gram_buf_start);
            A.op()->free(gram_buf);
            A.op()->free(A_buf);
            A.comm()->free_request(request_send);
            A.comm()->free_request(request_recv);
            Summary::end(METHOD_NAME);
            return gram;
        }
        error("Invalid input or not implemented yet.");
    }


    /**
     * @brief  Calculate \f$ \bm{\mathcal{A}} \times_n \bm{M} \f$, where
     * \f$ \bm{\mathcal{A}} \f$ is a tensor and \f$ \bm{M} \f$ is a matrix.
     *
     * @tparam Ty
     * @param A A matrix of shape \f$ I_1 \times \cdots \times I_N \f$
     * @param M A matrix of shape \f$ J_n \times I_n \f$
     * @param n Index for TTM routine.
     * @return Tensor<Ty>
     */
    template<typename Ty>
    Tensor<Ty> ttm(const Tensor<Ty> &A, const Tensor<Ty> &M, size_t n) {
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock &&
            (M.distribution() == nullptr ||
             M.distribution()->type() == Distribution::Type::kGlobal)) {
            // Initialization
            // Summary开始记录
            Summary::start(METHOD_NAME);
            // 类型要求：A.type = tensor, M.type = matrix
            assert(M.is_matrix());
            assert(A.shape_global()[n] == M.shape()[1]);
            // distrib 获得A的网格
            auto distrib = (DistributionCartesianBlock *) A.distribution();
            shape_t coord = distrib->coordinate();
            shape_t par = distrib->partition();
            size_t row_length = M.shape()[0];
            size_t col_length = M.shape()[1]; // M is of shape row_length * col_length.
            // M.size = (row_length, col_length)
            size_t col_local = A.shape()[n];
            size_t col_begin = DIANA_CEILDIV(col_length * coord[n], par[n]);
            size_t col_end = DIANA_CEILDIV(col_length * (coord[n] + 1),
                                           par[n]);
            assert(col_end - col_begin == col_local);
            size_t remain_size = A.size() / col_local;
            Ty *data_A = A.data();
            Ty *data_M = M.data();
            Ty *data_B = A.op()->alloc(A.size());
            Ty *data_Anew = A.op()->alloc(row_length * remain_size);
            // Matricization
            A.op()->tenmatt(data_B, data_A, A.shape(), n);
            // Do TTM
            A.op()->matmulNT(data_Anew, data_B,
                             data_M + col_begin * row_length,
                             remain_size, row_length, col_local);
            // Split communicator
            MPI_Comm comm_fiber = distrib->process_fiber_comm(n);
            // Do reduce-scatter
            shape_t new_shape = A.shape_global();
            new_shape[n] = row_length;
            Tensor<Ty> ret(A.distribution(), new_shape, false);
            Ty *data_ret = ret.data();
            Ty *data_ret_buf = ret.op()->alloc(ret.size());
            auto *recvcounts = Operator<int>::alloc(par[n]);
            for (size_t i = 0; i < par[n]; i++) {
                recvcounts[i] =
                        (int) DIANA_CEILDIV(row_length * (i + 1), par[n]) -
                        (int) DIANA_CEILDIV(row_length * i, par[n]);
                recvcounts[i] *= (int) remain_size;
            }
            ret.comm()->reduce_scatter(data_Anew, data_ret_buf, recvcounts,
                                       MPI_SUM, comm_fiber);
            // Tensorization
            ret.op()->mattten(data_ret, data_ret_buf, ret.shape(), n);
            // Free spaces
            A.op()->free(data_B);
            A.op()->free(data_Anew);
            A.op()->free(data_ret_buf);
            Operator<int>::free(recvcounts);
            Summary::end(METHOD_NAME);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty>
    ttmc(const Tensor<Ty> &A, const std::vector<Tensor<Ty>> &M,
         const std::vector<size_t> &idx) {
        Summary::start(METHOD_NAME);
        assert(M.size() == idx.size());
        for (size_t i = 0; i < M.size(); i++) {
            A = ttm(A, M[i], idx[i]);
        }
        Summary::end(METHOD_NAME);
        return A;
    }

    template<typename Ty>
    Tensor<Ty> gather(const Tensor<Ty> &A) {
        if (A.distribution() == nullptr) {
            error("This tensor is not distributed, cannot be gathered.");
        }
        if (A.distribution()->type() == Distribution::Type::kLocal) {
            error("Distribution of this tensor is Distribution::Type::kLocal, "
                  "cannot be gathered.");
        }
        if (A.distribution()->type() == Distribution::Type::kGlobal) {
            error("Distribution of this tensor is Distribution::Type::kGlobal, "
                  "there is no need to be gathered.");
        }
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            const int kZERO = 0;
            Tensor<Ty> ret(A.shape_global(), false);
            if (A.comm()->rank() == kZERO) {
                const int kMPISize = mpi_size();
                int *recvcounts = new int[(size_t) kMPISize];
                int *displs = new int[(size_t) kMPISize];
                // Calculate recvcounts
                for (int i = 0; i < kMPISize; i++) {
                    recvcounts[i] =
                            (int) A.distribution()->local_size(i,
                                                               A.shape_global());
                }
                // Calculate displs
                displs[0] = 0;
                for (int i = 1; i < kMPISize; i++) {
                    displs[i] = displs[i - 1] + recvcounts[i - 1];
                }
                // Receive data
                A.comm()->gatherv(A.data(), (int) A.size(), ret.data(),
                                  recvcounts,
                                  displs, kZERO);
                // Reorder data
                ret.op()->reorder_from_gather_cartesian_block(
                        ret.data(), ret.shape(),
                        ((DistributionCartesianBlock *) A.distribution())->partition(),
                        displs);
                // Bcast data
                A.comm()->bcast(ret.data(), (int) ret.size(), kZERO);
            } else {
                // Send data
                A.comm()->gatherv(A.data(), (int) A.size(), nullptr,
                                  nullptr,
                                  nullptr, kZERO);
                // Receive data from Bcast
                A.comm()->bcast(ret.data(), (int) ret.size(), kZERO);
            }
            Summary::end(METHOD_NAME);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    Tensor<Ty>
    scatter(const Tensor<Ty> &A, Distribution *distribution, int proc) {
        if (A.distribution() != nullptr &&
            A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            error("Distribution of this tensor is "
                  "Distribution::Type::kCartesianBlock,  there is no need to be "
                  "scatterd.");
        }
        if (A.distribution() != nullptr &&
            A.distribution()->type() == Distribution::Type::kGlobal) {
            error("Distribution of this tensor is Distribution::Type::kGlobal, "
                  "there is no need to be scatterd.");
        }
        if ((A.distribution() == nullptr ||
             A.distribution()->type() == Distribution::Type::kLocal) &&
            distribution->type() == Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            Tensor<Ty> ret(distribution, A.shape(), false);
            if (mpi_rank() == proc) {
                const int kMPISize = mpi_size();
                int *sendcounts = new int[(size_t) kMPISize];
                int *displs = new int[(size_t) kMPISize];
                // Calculate sendcounts
                for (int i = 0; i < kMPISize; i++) {
                    sendcounts[i] = (int) distribution->local_size(i,
                                                                   A.shape());
                }
                // Calculate displs
                displs[0] = 0;
                for (int i = 1; i < kMPISize; i++) {
                    displs[i] = displs[i - 1] + sendcounts[i - 1];
                }
                // Reorder data
                A.op()->reorder_for_scatter_cartesian_block(
                        A.data(), A.shape(),
                        ((DistributionCartesianBlock *) distribution)->partition(),
                        displs);
                // Scatter data
                ret.comm()->scatterv(A.data(), sendcounts, displs,
                                     ret.data(),
                                     (int) ret.size(), proc);
                // Reorder back data
                A.op()->reorder_from_gather_cartesian_block(
                        A.data(), A.shape(),
                        ((DistributionCartesianBlock *) distribution)->partition(),
                        displs);
            } else {
                // Receive data
                ret.comm()->scatterv(nullptr, nullptr, nullptr, ret.data(),
                                     (int) ret.size(), proc);
            }
            Summary::end(METHOD_NAME);
            return ret;
        }
        error("Invalid input or not implemented yet.");
    }

    template<typename Ty>
    double fnorm(const Tensor<Ty> &A) {
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            double ret = A.op()->fnorm(A.data(), A.size());
            ret = ret * ret;
            A.comm()->allreduce_inplace(&ret, 1, MPI_SUM);
//            A.comm()->allreduce_inplace(&ret, 1, MPI_SUM);
            Summary::end(METHOD_NAME);
            return sqrt(ret);
        } else {
            return A.op()->fnorm(A.data(), A.size());
        }
    }

    template<typename Ty>
    Ty sum(const Tensor<Ty> &A) {
        if (A.distribution()->type() ==
            Distribution::Type::kCartesianBlock) {
            Summary::start(METHOD_NAME);
            Ty ret = A.op()->sum(A.data(), A.size());
            A.comm()->allreduce_inplace(&ret, 1, MPI_SUM);
            Summary::end(METHOD_NAME);
            return ret;
        } else {
            return A.op()->sum(A.data(), A.size());
        }
    }

} // namespace Function