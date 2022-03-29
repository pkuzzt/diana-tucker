//
// Created by 30250 on 2022/3/17.
//
#include "algorithm.hpp"
#include "communicator.hpp"
#include "read_data.hpp"
#include "sp_tensor_mpi.tpp"
#include "sp_tensor.hpp"
#include "distribution.hpp"
#include <cstdio>

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    Summary::init();
    srand((unsigned int) 1);
    auto rank = mpi_rank();
    auto dbf = (data_buffer<double>*) malloc(sizeof(data_buffer<double>));
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: ./sptucker [filepath]\n");
        }
        exit(1);
    }
    else {
        read_data(argv[1], dbf);
    }
    SpTensor<double> A(dbf);
    free(dbf);
    shape_t par;
    if (mpi_size() == 1)
        par = {1, 1, 1};
    else
        par = {2, 2, 2};

    auto *distribution =
            new DistributionCartesianBlock(par, mpi_rank());
    shape_t R{10, 10, 10};
    auto[G, U] = Algorithm::SpTucker::Sp_HOOI_ALS(A, R, 5 ,distribution);
    Summary::finalize();
    Summary::print_summary();
    MPI_Finalize();
}