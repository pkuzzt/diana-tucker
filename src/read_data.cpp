//
// Created by 30250 on 2022/3/17.
//
#include "read_data.hpp"
#include "communicator.hpp"
#include "def.hpp"
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstdio>
#include <algorithm>

size_t read_int(const char* cbf, size_t &i) {
    char str[20];
    int j = 0;
    while (cbf[i] >= '0' && cbf[i] <= '9') {
        str[j++] = cbf[i++];
    }
    str[j] = '\0';
    i++;
    return (size_t) atoi(str);
}

double read_double(const char *cbf, size_t &i) {
    char str[20];
    int j = 0;
    while ((cbf[i] >= '0' && cbf[i] <= '9') || cbf[i] == '.') {
        str[j++] = cbf[i++];
    }
    str[j] = '\0';
    i++;
    return atof(str);
}


template<>
void read_data(char* filename, data_buffer<double> *dbf) {
    auto rank = 0;
    auto size = 1;

    if (mpi_rank() == 0) {
        int infile = open(filename, O_RDONLY);
        auto *statbuf = (struct stat*) malloc(sizeof(struct stat));
        size_t filesize, nnz, k;
        size_t ndim = 0, i, j;

        if (!infile) {
            fprintf(stderr, "Error: Can not find file: %s\n", filename);
            exit(1);
        }
        stat(filename, statbuf);
        filesize = (size_t )statbuf->st_size;

        auto start_index = DIANA_CEILDIV(filesize * (size_t) rank, (size_t) size);
        auto end_index = DIANA_CEILDIV(filesize * (size_t) (rank + 1), (size_t) size);

        free(statbuf);
//    if (rank == 0) {
//        fprintf(stdout, "filename  : %s\n", filename);
//        fprintf(stdout, "size      : %lu\n", filesize);
//    }
        char* cbf = (char*) mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, infile, 0);
        if (cbf == MAP_FAILED) {
            exit(1);
        }

        if (rank != 0) {
            while (*(cbf + start_index - 1) != '\n') {
                start_index++;
            }
        }

        if (rank != size - 1) {
            while (*(cbf + end_index) != '\n') {
                end_index++;
            }
            end_index++;
        }

        i = start_index; nnz = 0;
        while (true) {
            char ch = cbf[i++];
            if (ch == ' ') {
                ndim++;
            }
            if (ch == '\n') {
                break;
            }
        }

        for (i = start_index; i < end_index; i++) {
            if (cbf[i] == '\n') {
                nnz++;
            }
        }

        dbf->ndim = ndim;
        dbf->nnz = nnz;
//    printf("nnz = %lu, rank = %d\n", nnz, rank);
        dbf->index_lists = (size_t**) malloc(sizeof(size_t*) * ndim);
        for (i = 0; i < ndim; i++) {
            dbf->index_lists[i] = (size_t*) malloc(sizeof(size_t) * nnz);
        }
        dbf->vals = (double*) malloc(sizeof(double) * nnz);

        k = start_index;
        for (i = 0; i < nnz; i++) {
            for (j = 0; j < ndim; j++) {
                dbf->index_lists[j][i] = read_int(cbf, k) - 1;
            }
            dbf->vals[i] = read_double(cbf, k);
        }
//
//    for (i = 0; i < 10; i++) {
//        printf("%lu %lu %lu: %lf\n", dbf->index_lists[0][i], dbf->index_lists[1][i], dbf->index_lists[2][i], dbf->vals[i]);
//    }
        dbf->dims = (size_t*) malloc(sizeof(size_t) * ndim);

        for (i = 0; i < ndim; i++) {
            size_t local_max = 0;
            for (j = 0; j < nnz; j++) {
                if (local_max < dbf->index_lists[i][j])
                    local_max = dbf->index_lists[i][j];
            }
//        printf("dim: %lu, local_max: %lu\n", i, local_max);
            dbf->dims[i] = local_max + 1;
        }
    }
}

