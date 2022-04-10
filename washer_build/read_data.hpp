//
// Created by 30250 on 2022/3/17.
//

#ifndef DIANA_TUCKER_READ_DATA_HPP
#define DIANA_TUCKER_READ_DATA_HPP
#include "def.hpp"

template<typename Ty>
class data_buffer {
public:
    size_t ** index_lists;
    size_t ndim;
    size_t nnz;
    Ty * vals;
    size_t *dims;
};
#endif //DIANA_TUCKER_READ_DATA_HPP

template<typename Ty>
void read_data(char* filename, data_buffer<Ty> *dbf);