//
// Created by 丁明朔 on 2022/1/22.
//

#include "communicator.hpp"
#include "tensor.hpp"
#include "distribution.hpp"
#include "logger.hpp"
#include "algorithm.hpp"
#include "summary.hpp"


int main() {
    mpi_init();
    Summary::init();
    srand((unsigned int) 20000905);
    shape_t I{500, 400, 300};
    shape_t R{100, 100, 100};
    shape_t par{3, 2, 1};
    auto *distribution =
            new DistributionCartesianBlock(par, mpi_rank());
    auto T = Tensor<double>(distribution, I);
    T.randn();
    auto[G, U] = Algorithm::Tucker::HOOI_ALS(T, R, 5);
    Summary::finalize();
    Summary::print_summary();
    MPI_Finalize();
    return 0;
}