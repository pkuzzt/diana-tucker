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
    Communicator<size_t>::barrier();
    shape_t par;
    if (A.ndim() == 3) {
        switch(mpi_size()) {
            case 1:
                par = {1, 1, 1};
                break;
            case 2:
                par = {2, 1, 1};
                break;
            case 4:
                par = {2, 2, 1};
                break;
            case 8:
                par = {2, 2, 2};
                break;
            case 16:
                par = {4, 2, 2};
                break;
            case 32:
                par = {4, 4, 2};
                break;
            case 64:
                par = {4, 4, 4};
                break;
            case 128:
                par = {8, 4, 4};
                break;
            case 256:
                par = {8, 8, 4};
                break;
            case 512:
                par = {8, 8, 8};
        }

        shape_t R{100, 100, 100};
        auto[G, U] = Algorithm::SpTucker::Sp_HOOI_ALS(A, R, 5);
    }
    else {
        switch(mpi_size()) {
            case 1:
                par = {1, 1, 1, 1};
                break;
            case 2:
                par = {2, 1, 1, 1};
                break;
            case 4:
                par = {2, 2, 1, 1};
                break;
            case 8:
                par = {2, 2, 2, 1};
                break;
            case 16:
                par = {2, 2, 2, 2};
                break;
            case 32:
                par = {4, 2, 2, 2};
                break;
            case 64:
                par = {4, 4, 2, 2};
                break;
            case 128:
                par = {4, 4, 4, 2};
                break;
            case 256:
                par = {4, 4, 4, 4};
                break;
            case 512:
                par = {8, 4, 4, 4};
        }

        shape_t R{16, 16, 16, 16};
        auto[G, U] = Algorithm::SpTucker::Sp_HOOI_ALS(A, R, 5);
    }
    Summary::finalize();
    Summary::print_summary();
    MPI_Finalize();
}