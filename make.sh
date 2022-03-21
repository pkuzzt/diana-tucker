if [ -d "./build" ]; then
    rm -rf build
fi
mkdir build

cd build || exit
cmake -DUSE_OPENMP=1 -DUSE_LAPACK=1 -DUSE_MPI=1 ..

make -j

cp diana-tucker ..