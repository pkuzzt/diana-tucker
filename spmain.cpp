#include "algorithm.hpp"
#include "communicator.hpp"
#include "read_data.hpp"
#include "sp_tensor_mpi.tpp"
#include "sp_tensor.hpp"
#include "distribution.hpp"
#include <cstdio>
#include <omp.h>
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

    size_t truncation = 16;
    char * pend;

    if (argc >= 3) {
        omp_set_num_threads((int) strtol(argv[2], &pend, 10));
    }

    if (argc == 4) {
        truncation = (size_t) strtol(argv[3], &pend, 10);
    }
    SpTensor<double> A(dbf);
    Communicator<size_t>::barrier();
    shape_t par;
    output("MPI Size   : " + std::to_string(mpi_size()));
    output(std::string("File path  : ") + argv[1]);
    output("truncation : " + std::to_string(truncation));
    output("nthread    : " + std::to_string(omp_get_max_threads()));
    shape_t R;
    for (size_t i = 0; i < A.ndim(); i++) {
        R.push_back(truncation);
    }
    auto[G, U] = Algorithm::SpTucker::Sp_HOOI_ALS(A, R, 5);

    Summary::finalize();
    Summary::print_summary();
    MPI_Finalize();
}
