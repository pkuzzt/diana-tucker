#ifndef __DIANA_CORE_INCLUDE_TENSOR_HPP__
#define __DIANA_CORE_INCLUDE_TENSOR_HPP__

#include "def.hpp"

#include "communicator.hpp"
#include "distribution.hpp"
#include "operator.hpp"

#include <map>

template<typename Ty>
class Tensor;

template<typename Ty>
const Tensor<Ty> operator+(const Tensor<Ty> &, const Tensor<Ty> &);

template<typename Ty>
const Tensor<Ty> operator-(const Tensor<Ty> &, const Tensor<Ty> &);

template<typename Ty>
const Tensor<Ty> operator*(const Tensor<Ty> &, const Tensor<Ty> &);

template<typename Ty>
const Tensor<Ty> operator*(Ty, const Tensor<Ty> &);

template<typename Ty>
const Tensor<Ty> operator*(const Tensor<Ty> &, Ty);

template<typename Ty>
const Tensor<Ty> operator/(const Tensor<Ty> &, Ty);

enum Transpose : int {
    kN,
    kT,
    kC,
};

/**
 * @brief Tensor class
 *
 * @tparam Ty
 */
template<typename Ty>
class Tensor {
private:
    Ty *data_;
    size_t ndim_;
    size_t size_;
    shape_t shape_;
    shape_t stride_;
    shape_t stride_in_bytes_;
    bool is_matrix_;
    Transpose trans_;
    Operator<Ty> *op_;

    size_t size_global_;
    shape_t shape_global_;
    Distribution *distribution_; /**< Distribution type of this tensor. */
    Communicator<Ty> *comm_;     /**< Communicator of this tensor. */
    int comm_size_;
    int comm_rank_;

    static std::map<Ty *, int> ref_count;

    inline void init_by_shape(const shape_t &);

    static inline void assert_shape(const Tensor<Ty> &, const Tensor<Ty> &);

    static inline void assert_permutation(const Tensor<Ty> &, const shape_t &);

    inline int shape_t_to_index(const shape_t &);

    inline void init_by_distribution(const shape_t &shape_global,
                                     Distribution *distribution);

public:
    Tensor();

    Tensor(Ty *, const shape_t &);

    Tensor(const shape_t &, bool = true);

    Tensor(Distribution *distribution, Ty *, const shape_t &);

    Tensor(Distribution *distribution, const shape_t &, bool = true);

    Tensor(const Tensor<Ty> &);

    ~Tensor();

    const Tensor<Ty> operator=(const Tensor<Ty> &);

    inline Ty *data() const;

    inline Operator<Ty> *op() const;

    inline size_t ndim() const;

    inline size_t size() const;

    inline const shape_t &stride() const;

    inline const shape_t &stride_in_bytes() const;

    inline const shape_t &shape() const;

    inline bool is_matrix() const;

    inline Transpose trans() const;

    inline Communicator<Ty> *comm() const;

    inline Distribution *distribution() const;

    inline int comm_size() const;

    inline int comm_rank() const;

    inline const shape_t &shape_global() const;

    inline size_t size_global() const;

    Tensor<Ty> gather();

    Tensor<Ty> scatter(Distribution *distribution, int proc);

    void sync(int proc);

    Ty &operator[](size_t);

    Ty &operator()(size_t, ...);

    Ty getitem(const shape_t &);

    Ty getitem(int);

    void setitem(const shape_t &, Ty);

    void setitem(int, Ty);

    void constant(Ty c);

    void zeros();

    void ones();

    void rand();

    void randn();

    void add(const Tensor<Ty> &);

    void sub(const Tensor<Ty> &);

    void mul(const Tensor<Ty> &);

    void nmul(Ty);

    void ndiv(Ty);

    void reshape(const shape_t &);

    void permute(const shape_t &);

    double fnorm() const;

    Ty sum() const;

    Tensor<Ty> copy() const;

    void print() const;

    template<typename T1>
    friend const Tensor<T1> operator+(const Tensor<T1> &, const Tensor<T1> &);

    template<typename T1>
    friend const Tensor<T1> operator-(const Tensor<T1> &, const Tensor<T1> &);

    template<typename T1>
    friend const Tensor<T1> operator*(const Tensor<T1> &, const Tensor<T1> &);
};

#include "tensor_mpi.tpp"

#endif