//
// Created by 丁明朔 on 2022/3/11.
//

#include "FunctionDistributedTest.hpp"

TEST_F(FunctionDistributedTest, TTM1) {
    // Initialization
    auto *dis_global = new DistributionGlobal();
    Tensor<double> m1(dis_global, {4, 4});
    for (size_t i = 0; i < m1.size(); i++) {
        m1[i] = (double) i;
    }
    Tensor<double> m2(dis_global, {2, 3});
    for (size_t i = 0; i < m2.size(); i++) {
        m2[i] = (double) i;
    }
    // Calculate
    t = Function::ttm<double>(t, m1, 1);
    t = Function::ttm<double>(t, m2, 2);
    // Gather
    auto ans = Function::gather(t);
    // Ground Truth
    if (mpi_rank() == 0) {
        double ground_truth[] = {4752.0, 4896.0, 5040.0, 5888.0, 6032.0, 5310.0,
                                 5478.0, 5646.0, 6620.0, 6788.0, 5868.0, 6060.0,
                                 6252.0, 7352.0, 7544.0, 6426.0, 6642.0, 6858.0,
                                 8084.0, 8300.0, 6960.0, 7176.0, 7392.0, 8720.0,
                                 8936.0, 7761.0, 8013.0, 8265.0, 9794.0,
                                 10046.0, 8562.0, 8850.0, 9138.0, 10868.0,
                                 11156.0, 9363.0, 9687.0, 10011.0, 11942.0,
                                 12266.0};
        for (size_t i = 0; i < ans.size(); i++) {
            EXPECT_DOUBLE_EQ(ans[i], ground_truth[i]);
        }
    }
}