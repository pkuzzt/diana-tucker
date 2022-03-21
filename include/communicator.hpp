#ifndef __DIANA_CORE_INCLUDE_COMMUNICATOR_HPP__
#define __DIANA_CORE_INCLUDE_COMMUNICATOR_HPP__

#include <cstdlib>
#include <mpi.h>
#include <vector>

void mpi_init();

int mpi_rank();

int mpi_size();

template<typename Ty>
class Communicator {
private:
    int rank_;
    int size_;

public:
    Communicator();

    ~Communicator();

    [[nodiscard]] int rank() const;

    [[nodiscard]] int size() const;

    [[nodiscard]] constexpr static inline MPI_Datatype mpi_type();

    [[nodiscard]]  static inline MPI_Request *new_request();

    static inline void free_request(MPI_Request *request);

    static MPI_Comm
    comm_split(int color, int rank, MPI_Comm comm = MPI_COMM_WORLD);

    static void
    bcast(Ty *A, int size, int proc, MPI_Comm comm = MPI_COMM_WORLD);

    static void allreduce_inplace(Ty *A, int size, MPI_Op op,
                                  MPI_Comm comm = MPI_COMM_WORLD);

    static void allreduce(Ty *sendbuf, Ty *recvbuf, int size, MPI_Op op,
                          MPI_Comm comm = MPI_COMM_WORLD);

    static void
    sendrecv(Ty *A, int size, int dest, MPI_Comm comm = MPI_COMM_WORLD);

    static void sendrecv(Ty *sendbuf, int sendcount, int dest, Ty *recvbuf,
                         int recvcount, int source,
                         MPI_Comm comm = MPI_COMM_WORLD);

    static void reduce_scatter(Ty *sendbuf, Ty *recvbuf, const int *recvcounts,
                               MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD);

    static void allgather(const Ty *sendbuf, int sendcount, Ty *recvbuf,
                          MPI_Comm comm = MPI_COMM_WORLD);

    static void allgatherv(const Ty *sendbuf, int sendcount, Ty *recvbuf,
                           const int *recvcounts, const int *displs,
                           MPI_Comm comm = MPI_COMM_WORLD);

    static void
    gatherv(Ty *sendbuf, int sendcount, Ty *recvbuf, const int *recvcounts,
            const int *displs, int root, MPI_Comm comm = MPI_COMM_WORLD);

    static void scatterv(Ty *sendbuf, const int *sendcounts, const int *displs,
                         Ty *recvbuf, int recvcount, int root,
                         MPI_Comm comm = MPI_COMM_WORLD);

    static void barrier(MPI_Comm comm = MPI_COMM_WORLD);

    static void wait(MPI_Request *request);

    static void isend(MPI_Request *request, Ty *buf, int count, int dest,
                      MPI_Comm comm = MPI_COMM_WORLD,
                      int tag = 0);

    static void send(Ty *buf, int count, int dest, MPI_Comm comm = MPI_COMM_WORLD,
                     int tag = 0, MPI_Status *status = MPI_STATUS_IGNORE);

    static void irecv(MPI_Request *request, Ty *buf, int count, int source,
                      MPI_Comm comm = MPI_COMM_WORLD,
                      int tag = 0);

    static void recv(Ty *buf, int count, int source, MPI_Comm comm = MPI_COMM_WORLD,
                     int tag = 0, MPI_Status *status = MPI_STATUS_IGNORE);
};

#include "communicator.tpp"

#endif