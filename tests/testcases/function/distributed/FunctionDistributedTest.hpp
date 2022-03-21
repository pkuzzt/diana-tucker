//
// Created by 丁明朔 on 2022/3/11.
//

#ifndef DIANA_TUCKER_FUNCTIONDISTRIBUTEDTEST_HPP
#define DIANA_TUCKER_FUNCTIONDISTRIBUTEDTEST_HPP

#include "tensor.hpp"
#include "gtest/gtest.h"

class FunctionDistributedTest : public ::testing::Test {
protected:
    void SetUp() override;

    Tensor<double> t;
};


#endif //DIANA_TUCKER_FUNCTIONDISTRIBUTEDTEST_HPP
