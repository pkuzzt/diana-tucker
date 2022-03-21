#include "def.hpp"
#include "logger.hpp"
#include "summary.hpp"
#include <cstdint>
#include <climits>

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#endif

template<class Ty>
Communicator<Ty>::Communicator() {
    MPI_Comm_size(MPI_COMM_WORLD, &this->size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
}

template<class Ty>
Communicator<Ty>::~Communicator() = default;

template<class Ty>
int Communicator<Ty>::size() const { return this->size_; }

template<class Ty>
int Communicator<Ty>::rank() const { return this->rank_; }

template<class Ty>
constexpr MPI_Datatype Communicator<Ty>::mpi_type() {
    if constexpr (std::is_same<Ty, float32>::value) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same<Ty, float64>::value) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same<Ty, complex32>::value) {
        return MPI_C_COMPLEX;
    } else if constexpr (std::is_same<Ty, complex64>::value) {
        return MPI_C_DOUBLE_COMPLEX;
    } else if constexpr (std::is_same<Ty, int>::value) {
        return MPI_INT;
    } else if constexpr (std::is_same<Ty, long long>::value) {
        return MPI_LONG_LONG_INT;
    } else if constexpr (std::is_same<Ty, size_t>::value) {
        return MPI_SIZE_T;
    } else {
        error("Invalid type.");
    }
}

template<class Ty>
MPI_Request *Communicator<Ty>::new_request() {
    return new MPI_Request;
}

template<class Ty>
void Communicator<Ty>::free_request(MPI_Request *request) {
    delete request;
}

template<class Ty>
MPI_Comm Communicator<Ty>::comm_split(int color, int rank, MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Comm ret;
    MPI_Comm_split(comm, color, rank, &ret);
    Summary::end(METHOD_NAME);
    return ret;
}

template<class Ty>
void Communicator<Ty>::bcast(Ty *A, int size, int proc, MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Bcast(A, size, mpi_type(), proc, comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::allreduce_inplace(Ty *A, int size, MPI_Op op,
                                         MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Allreduce(MPI_IN_PLACE, A, size, mpi_type(), op, comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::allreduce(Ty *sendbuf, Ty *recvbuf, int size, MPI_Op op,
                                 MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Allreduce(sendbuf, recvbuf, size, mpi_type(), op, comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::sendrecv(Ty *A, int size, int des, MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Status status;
    MPI_Sendrecv_replace(A, size, mpi_type(), des, 0, des, 0, comm,
                         &status);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void
Communicator<Ty>::sendrecv(Ty *sendbuf, int sendcount, int dest, Ty *recvbuf,
                           int recvcount, int source, MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Status status;
    MPI_Sendrecv(sendbuf, sendcount, mpi_type(), dest, 0, recvbuf,
                 recvcount, mpi_type(), source, 0, comm, &status);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::reduce_scatter(Ty *sendbuf, Ty *recvbuf,
                                      const int *recvcounts, MPI_Op op,
                                      MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, mpi_type(), op,
                       comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void
Communicator<Ty>::allgather(const Ty *sendbuf, int sendcount, Ty *recvbuf,
                            MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Allgather(sendbuf, sendcount, mpi_type(), recvbuf,
                  sendcount, mpi_type(), comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void
Communicator<Ty>::allgatherv(const Ty *sendbuf, int sendcount, Ty *recvbuf,
                             const int *recvcounts, const int *displs,
                             MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Allgatherv(sendbuf, sendcount, mpi_type(), recvbuf, recvcounts, displs,
                   mpi_type(), comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::gatherv(Ty *sendbuf, int sendcount, Ty *recvbuf,
                               const int *recvcounts, const int *displs,
                               int root, MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Gatherv(sendbuf, sendcount, mpi_type(), recvbuf, recvcounts,
                displs, mpi_type(), root, comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::scatterv(Ty *sendbuf, const int *sendcounts,
                                const int *displs, Ty *recvbuf, int recvcount,
                                int root, MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Scatterv(sendbuf, sendcounts, displs, mpi_type(), recvbuf,
                 recvcount, mpi_type(), root, comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::barrier(MPI_Comm comm) {
    Summary::start(METHOD_NAME);
    MPI_Barrier(comm);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void Communicator<Ty>::wait(MPI_Request *request) {
    Summary::start(METHOD_NAME);
    MPI_Status status;
    MPI_Wait(request, &status);
    Summary::end(METHOD_NAME);
}

template<class Ty>
void
Communicator<Ty>::isend(MPI_Request *request, Ty *buf, int count, int dest,
                        MPI_Comm comm, int tag) {
    MPI_Isend(buf, count, mpi_type(), dest, tag, comm, request);
}

template<class Ty>
void
Communicator<Ty>::send(Ty *buf, int count, int dest, MPI_Comm comm,
                       int tag, MPI_Status *status) {
    MPI_Send(buf, count, mpi_type(), dest, tag, comm, status);
};

template<class Ty>
void
Communicator<Ty>::irecv(MPI_Request *request, Ty *buf, int count, int source,
                        MPI_Comm comm,
                        int tag) {
    MPI_Irecv(buf, count, mpi_type(), source, tag, comm, request);
}

template<class Ty>
void
Communicator<Ty>::recv(Ty* buf, int count, int source, MPI_Comm comm,
                       int tag, MPI_Status *status) {
    MPI_Recv(buf, count, mpi_type(), source, tag, comm, status);
}