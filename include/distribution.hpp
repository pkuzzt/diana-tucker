#ifndef __DIANA_CORE_DISTRIBUTION_TENSOR_HPP__
#define __DIANA_CORE_DISTRIBUTION_TENSOR_HPP__

#include "def.hpp"
#include <mpi.h>

/**
 * @enum Distribution
 * @brief Enumerate class of tensor's distribution.
 *
 * Distribution enumerate class using to indicate multiple types of tensor
 * distribution.
 */
class Distribution {
public:
    enum Type : int {
        kLocal,  /**< this tensor is only stored on local process. */
        kGlobal, /**< a redundant copy of this tensor is stored on each process.
                  */
        kCartesianBlock, /**< this tensor is blockly stored on cartesian
                            processes. */
    };

private:
    Type type_;

public:
    Distribution() = delete;

    explicit Distribution(Distribution::Type type);

    [[nodiscard]] Distribution::Type type() const;

    virtual void
    get_local_data(const shape_t &global_shape, shape_t &local_shape,
                   shape_t &local_start, shape_t &local_end);

    virtual void
    get_local_shape(const shape_t &global_shape, shape_t &local_shape);

    virtual size_t global_size(const shape_t &global_shape);

    virtual size_t local_size(const shape_t &global_shape);

    virtual size_t local_size(int rank, const shape_t &global_shape);
};

class DistributionLocal : public Distribution {
public:
    DistributionLocal();
};

class DistributionGlobal : public Distribution {
public:
    DistributionGlobal();
};

class DistributionCartesianBlock : public Distribution {
private:
    shape_t partition_;
    shape_t coordinate_;
    size_t ndim_;
    std::vector<MPI_Comm> process_fiber_comm_;

public:
    DistributionCartesianBlock(shape_t partition, int rank);

    [[nodiscard]] shape_t partition() const;

    [[nodiscard]] shape_t coordinate() const;

    [[nodiscard]] shape_t coordinate(int rank) const;

    [[nodiscard]] size_t ndim() const;

    void get_local_data(const shape_t &global_shape, shape_t &local_shape,
                        shape_t &local_start,
                        shape_t &local_end) override;

    void
    get_local_shape(const shape_t &global_shape, shape_t &local_shape) override;

    size_t local_size(const shape_t &global_shape) override;

    size_t local_size(int rank, const shape_t &global_shape) override;

    std::tuple<int, int> process_fiber(size_t n);

    MPI_Comm process_fiber_comm(size_t n);
};

#endif