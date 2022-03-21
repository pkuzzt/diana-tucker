#ifndef __DIANA_CORE_INCLUDE_QUANTUM_QREGISTER_HPP__
#define __DIANA_CORE_INCLUDE_QUANTUM_QREGISTER_HPP__

#include "quantum/circuit.hpp"
#include "quantum/gate.hpp"
#include "communicator.hpp"
#include "tensor.hpp"
#include "def.hpp"
#include <vector>
#include <tuple>

class QRegister {
  private:
    int qubit_number_;

  public:
    QRegister(int);
    ~QRegister();

    int qubit_number();

    virtual complex64 amplitude(long long) = 0;
    virtual float64 probability(long long) = 0;

    virtual void apply_gate(const Gate &) = 0;
    virtual void apply_circuit(const Circuit &) = 0;

    virtual void print() = 0;
};

class SchrodingerQRegister : public QRegister {
  private:
    Tensor<complex64> state_vector_;
    CommunicatorBlock1D<complex64> comm_;
    int comm_rank_;
    int comm_size_;

    std::vector<Tensor<complex64>> generate_p();
    std::vector<Tensor<complex64>> partition(Tensor<complex64>);
    void apply_single_gate(const Gate &);
    void apply_controlled_gate(const Gate &);
    void apply_double_gate(const Gate &);

  public:
    SchrodingerQRegister(int);
    ~SchrodingerQRegister();

    virtual complex64 amplitude(long long);
    virtual float64 probability(long long);

    virtual void apply_gate(const Gate &);
    virtual void apply_circuit(const Circuit &);

    virtual void print();
};

class SubspaceCPQRegister : public QRegister {
  private:
    CommunicatorBlock1D<complex64> comm_;
    int comm_rank_;
    int comm_size_;

    std::vector<Tensor<complex64>> factor_;
    Tensor<complex64> lambda_;
    vint par_;
    int p_;
    int rank_;
    Tensor<complex64> factor_changed_[2][4];
    int factor_changed_idx_[2];
    int factor_changed_len_[2];

    std::vector<Tensor<complex64>> generate_p();
    std::vector<Tensor<complex64>> partition(Tensor<complex64>);
    std::tuple<int, int> get_index(int);
    void apply_single_gate(const Gate &);
    void apply_controlled_gate(const Gate &);
    void apply_double_gate(const Gate &);
    void compress(float64);

  public:
    SubspaceCPQRegister(int, vint);
    ~SubspaceCPQRegister();

    virtual complex64 amplitude(long long);
    virtual float64 probability(long long);

    virtual void apply_gate(const Gate &);
    virtual void apply_circuit(const Circuit &, float64);

    virtual void print();
};

#endif