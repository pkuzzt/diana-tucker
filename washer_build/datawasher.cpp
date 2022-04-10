//
// Created by 30250 on 2022/3/30.
//

#include "read_data.hpp"
#include <cstdlib>
#include <cstdio>

int main(int argc, char * argv[]) {
    auto dbf = (data_buffer<double>*) malloc(sizeof(data_buffer<double>));
    read_data(argv[1], dbf);
    auto outfile = fopen(argv[2], "w");
    auto mid = dbf->nnz / 2;
    for (size_t i = 0; i < mid; i++) {
        for (size_t j = 0; j < dbf->ndim; j++)
            fprintf(outfile, "%lu ", dbf->index_lists[j][i] + 1);
        fprintf(outfile, "%lf\n", dbf->vals[i]);

        for (size_t j = 0; j < dbf->ndim; j++)
            fprintf(outfile, "%lu ", dbf->index_lists[j][i + mid] + 1);
        fprintf(outfile, "%lf\n", dbf->vals[i + mid]);
    }
    for (size_t i = 2 * mid; i < dbf->nnz; i++) {
        for (size_t j = 0; j < dbf->ndim; j++)
            fprintf(outfile, "%lu ", dbf->index_lists[j][i] + 1);
        fprintf(outfile, "%lf\n", dbf->vals[i]);
    }
    free(dbf);
}