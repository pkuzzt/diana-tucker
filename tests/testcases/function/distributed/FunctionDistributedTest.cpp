//
// Created by 丁明朔 on 2022/3/11.
//

#include "FunctionDistributedTest.hpp"

void FunctionDistributedTest::SetUp() {
    shape_t shape{5, 4, 3};
    shape_t par{2, 3, 1};
    auto *distribution =
            new DistributionCartesianBlock(par, mpi_rank());
    t = Tensor<double>(distribution, shape);
    for (size_t i = 0; i < t.size(); i++) {
        t[i] = 10.0 * mpi_rank() + 1.0 * (double) i;
    }
}