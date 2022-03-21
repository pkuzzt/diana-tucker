#ifndef __DIANA_CORE_INCLUDE_UTIL_HPP__
#define __DIANA_CORE_INCLUDE_UTIL_HPP__

#include "def.hpp"

namespace Util {
void memcpy(void *, void *, size_t);

double randn();
size_t calc_size(const shape_t &shape);
shape_t calc_stride(const shape_t &shape);
}; // namespace Util

class RangeIter {
  private:
    int value_;

  public:
    RangeIter(int);
    ~RangeIter();

    int value() const;

    bool operator!=(const RangeIter &) const;
    int operator*() const;
    const RangeIter &operator++();
};

class Range {
  private:
    int begin_value_;
    int end_value_;

  public:
    Range(int);
    Range(int, int);
    ~Range() {}

    RangeIter begin() const;
    RangeIter end() const;
};

#endif