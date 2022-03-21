#include "distribution.hpp"
#include "def.hpp"
#include "logger.hpp"
#include "communicator.hpp"
#include <tuple>

void distribution_assert_valid_input_(const shape_t &global_shape,
                                      const shape_t &local_shape,
                                      const shape_t &local_start = shape_t(),
                                      const shape_t &local_end = shape_t()) {
#ifndef PERFORMANCE_MODE
    assert(!global_shape.empty());
    assert(local_shape.empty());
    assert(local_start.empty());
    assert(local_end.empty());
    for (auto dim: global_shape) {
        assert(dim > 0);
    }
#endif
}

Distribution::Distribution(Distribution::Type type) { this->type_ = type; }

Distribution::Type Distribution::type() const { return this->type_; }

void
Distribution::get_local_data(const shape_t &global_shape, shape_t &local_shape,
                             shape_t &local_start, shape_t &local_end) {
    distribution_assert_valid_input_(global_shape, local_shape, local_start,
                                     local_end);
    for (auto dim: global_shape) {
        local_shape.push_back(dim);
        local_start.push_back(0);
        local_end.push_back(dim);
    }
}

void Distribution::get_local_shape(const shape_t &global_shape,
                                   shape_t &local_shape) {
    distribution_assert_valid_input_(global_shape, local_shape);
    for (auto dim: global_shape) {
        local_shape.push_back(dim);
    }
}

size_t Distribution::global_size(const shape_t &global_shape) {
    size_t size = 1;
    for (size_t dim: global_shape) {
        size *= dim;
    }
    return size;
}

size_t Distribution::local_size(const shape_t &local_shape) {
    size_t size = 1;
    for (size_t dim: local_shape) {
        size *= dim;
    }
    return size;
}

size_t Distribution::local_size(int rank, const shape_t &local_shape) {
    DIANA_UNUSED(rank);
    return Distribution::local_size(local_shape);
}

DistributionLocal::DistributionLocal()
        : Distribution(Distribution::Type::kLocal) {}

DistributionGlobal::DistributionGlobal()
        : Distribution(Distribution::Type::kGlobal) {}

DistributionCartesianBlock::DistributionCartesianBlock(shape_t partition,
                                                       int rank)
        : Distribution(Distribution::Type::kCartesianBlock) {
    this->ndim_ = partition.size();
    this->partition_.assign(partition.begin(), partition.end());
    this->coordinate_ = shape_t();
    for (auto item: partition) {
        this->coordinate_.push_back((size_t) rank % item);
        rank /= (int) item;
    }
    assert(this->ndim_ == this->coordinate_.size());
    for (size_t i = 0; i < this->ndim_; i++) {
        assert(this->coordinate_[i] < this->partition_[i]);
        assert(this->partition_[i] > 0);
    }
    for (size_t n = 0; n < this->ndim_; n++) {
        auto[new_color, new_rank] = this->process_fiber(n);
        this->process_fiber_comm_.push_back(
                Communicator<void>::comm_split(new_color, new_rank));
    }
}

shape_t DistributionCartesianBlock::partition() const {
    return this->partition_;
}

shape_t DistributionCartesianBlock::coordinate() const {
    return this->coordinate_;
}

shape_t DistributionCartesianBlock::coordinate(int rank) const {
    shape_t coordinate;
    for (auto item: this->partition_) {
        coordinate.push_back((size_t) rank % item);
        rank /= (int) item;
    }
    return coordinate;
}

size_t DistributionCartesianBlock::ndim() const { return this->ndim_; }

void DistributionCartesianBlock::get_local_data(const shape_t &global_shape,
                                                shape_t &local_shape,
                                                shape_t &local_start,
                                                shape_t &local_end) {
    distribution_assert_valid_input_(global_shape, local_shape, local_start,
                                     local_end);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * this->coordinate_[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (this->coordinate_[i] + 1),
                                   this->partition_[i]);
        local_start.push_back(start);
        local_end.push_back(end);
        local_shape.push_back(end - start);
    }
}

void DistributionCartesianBlock::get_local_shape(const shape_t &global_shape,
                                                 shape_t &local_shape) {
    distribution_assert_valid_input_(global_shape, local_shape);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * this->coordinate_[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (this->coordinate_[i] + 1),
                                   this->partition_[i]);
        local_shape.push_back(end - start);
    }
}

size_t DistributionCartesianBlock::local_size(const shape_t &global_shape) {
    size_t size = 1;
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * this->coordinate_[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (this->coordinate_[i] + 1),
                                   this->partition_[i]);
        size *= (end - start);
    }
    return size;
}

size_t DistributionCartesianBlock::local_size(int rank,
                                              const shape_t &global_shape) {
    size_t size = 1;
    auto coord = DistributionCartesianBlock::coordinate(rank);
    for (size_t i = 0; i < this->ndim_; i++) {
        size_t start = DIANA_CEILDIV(global_shape[i] * coord[i],
                                     this->partition_[i]);
        size_t end = DIANA_CEILDIV(global_shape[i] * (coord[i] + 1),
                                   this->partition_[i]);
        size *= (end - start);
    }
    return size;
}

/**
 * Return new process color and process rank when get the n-th process fiber.
 * @param n
 * @return color, rank
 */
std::tuple<int, int> DistributionCartesianBlock::process_fiber(size_t n) {
    int new_rank = (int) this->coordinate_[n];
    int new_color = 0;
    int pre = 1;
    for (size_t d = 0; d < this->ndim_; d++) {
        if (d != n) {
            new_color += (int) this->coordinate_[d] * pre;
            pre *= (int) this->partition_[d];
        }
    }
    return std::make_tuple(new_color, new_rank);
}

MPI_Comm DistributionCartesianBlock::process_fiber_comm(size_t n) {
    return this->process_fiber_comm_[n];
}