//
// Created by 丁明朔 on 2022/3/11.
//

#include "FunctionDistributedTest.hpp"

TEST_F(FunctionDistributedTest, Gram1) {
// Calculate
    auto ans = Function::gram<double>(t, 1);
// Ground Truth
    if (mpi_rank() == 0) {
        double ground_truth[] = {1990.0, 2353.0, 4487.0, 7487.0, 2353.0, 2821.0,
                                 5525.0, 9305.0, 4487.0, 5525.0, 11599.0,
                                 19819.0, 7487.0, 9305.0, 19819.0, 34039.0};
        for (size_t i = 0; i < ans.size(); i++) {
            EXPECT_DOUBLE_EQ(ans[i], ground_truth[i]);
        }
    }
}