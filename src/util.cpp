#include "util.hpp"

#include <cmath>
#include <cstring>

void Util::memcpy(void *dst, void *src, size_t len) {
    std::memcpy(dst, src, len);
}

double Util::randn() {
    double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1)
        return Util::randn();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

size_t Util::calc_size(const shape_t &shape) {
    size_t ret = 1;
    size_t ndim = shape.size();
    for (size_t i = 0; i < ndim; i++) {
        ret *= shape[i];
    }
    return ret;
}

shape_t Util::calc_stride(const shape_t &shape) {
    shape_t stride;
    stride.push_back(1);
    size_t ndim = shape.size();
    for (size_t i = 1; i < ndim; i++) {
        stride.push_back(shape[i - 1] * shape[i - 1]);
    }
    return stride;
}

int RangeIter::value() const { return this->value_; }

RangeIter::RangeIter(int val) { this->value_ = val; }

bool RangeIter::operator!=(const RangeIter &other) const {
    return (this->value()) != (other.value());
}

int RangeIter::operator*() const { return value(); }

const RangeIter &RangeIter::operator++() {
    ++this->value_;
    return *this;
}

Range::Range(int end_v) {
    this->begin_value_ = 0;
    this->end_value_ = end_v;
}

Range::Range(int begin_v, int end_v) {
    this->begin_value_ = begin_v;
    this->end_value_ = end_v;
}

RangeIter Range::begin() const { return RangeIter(this->begin_value_); }

RangeIter Range::end() const { return RangeIter(this->end_value_); }