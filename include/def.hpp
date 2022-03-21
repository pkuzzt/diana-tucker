#ifndef __DIANA_CORE_SRC_INCLUDE_DEF_HPP__
#define __DIANA_CORE_SRC_INCLUDE_DEF_HPP__

#include <cstdlib>

#include <vector>
typedef std::vector<int> vint;
typedef std::vector<size_t> shape_t;

#include <complex>
typedef float float32;
typedef double float64;
typedef std::complex<float> complex32;
typedef std::complex<double> complex64;

//#include <pybind11/numpy.h>
// namespace py = pybind11;

#include <cstddef>

namespace Constant {
// tensor.hpp
const int kMaxPrintLength = 6;
const int kPrintPrecision = 4;
}; // namespace Constant

#define DIANA_CEILDIV(n, k) (((n) + (k)-1) / (k))
#define DIANA_UNUSED(x) (void)x;

#endif