#ifndef __DIANA_CORE_INCLUDE_QUANTUM_GATE_HPP__
#define __DIANA_CORE_INCLUDE_QUANTUM_GATE_HPP__

#include "def.hpp"
#include "tensor.hpp"

#include <string>

enum class GateType {
    kArbitary,
    kSingle,
    kDouble,
    kControlled,
    kSwap,
};

extern const char *GATE_NAME[4];

class Gate {
  private:
    GateType type_;
    int size_;
    vint qubit_id_;
    Tensor<complex64> weight_;
    std::string name_;

  public:
    Gate(GateType, int, Tensor<complex64> &); /**< Single qubit gate. */
    Gate(GateType, int, int,
         Tensor<complex64> &); /**< Double and controlled qubit gate. */
    Gate(GateType, vint, Tensor<complex64> &); /**< Arbitarty qubit gate. */
    Gate(const Gate &g);
    ~Gate();

    int size() const;
    GateType type() const;
    vint qubit_id() const;
    Tensor<complex64> weight() const;
    std::string name() const;

    void set_name(std::string);
};

#endif
