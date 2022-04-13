#!/bin/bash
module purge
module load gcc/9.3.0
module load intel/2018.1
module load cmake/3.16.0
export CC=/gpfs/share/software/gcc/9.3.0/bin/gcc
export CXX=/gpfs/share/software/gcc/9.3.0/bin/g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/share/software/gcc/9.3.0/lib64