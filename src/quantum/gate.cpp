#include "quantum/gate.hpp"
#include "logger.hpp"

#include <iostream>


Gate::Gate(GateType type, int qubit_id, Tensor<complex64> &weight) {
    assert(type == GateType::kSingle);
    assert(weight.ndim() == 2);
    assert(weight.shape()[0] == 2 && weight.shape()[1] == 2);

    this->type_ = type;
    this->size_ = 1;
    this->qubit_id_ = vint();
    this->qubit_id_.push_back(qubit_id);
    this->weight_ = weight;
}

Gate::Gate(GateType type, int qubit_id_x, int qubit_id_y,
           Tensor<complex64> &weight) {
    assert(type == GateType::kDouble || type == GateType::kControlled ||
           type == GateType::kSwap);
    assert(weight.ndim() == 2);
    assert(weight.shape()[0] == 4 && weight.shape()[1] == 4);

    this->type_ = type;
    this->size_ = 2;
    this->qubit_id_ = vint();
    this->qubit_id_.push_back(qubit_id_x);
    this->qubit_id_.push_back(qubit_id_y);
    this->weight_ = weight;
}

Gate::Gate(GateType type, vint qubit_id, Tensor<complex64> &weight) {
    assert(weight.ndim() == 2);
    assert((type == GateType::kSingle && weight.shape()[0] == 2 &&
            weight.shape()[1] == 2) ||
           ((type == GateType::kDouble || type == GateType::kControlled ||
             type == GateType::kSwap) &&
            weight.shape()[0] == 4 && weight.shape()[1] == 4) ||
           (type == GateType::kArbitary &&
            weight.shape()[0] == (1 << qubit_id.size()) &&
            weight.shape()[1] == (1 << qubit_id.size())));

    this->type_ = type;
    this->size_ = qubit_id.size();
    this->qubit_id_ = vint();
    this->qubit_id_.assign(qubit_id.begin(), qubit_id.end());
    this->weight_ = weight;
}

Gate::Gate(const Gate &g) {
    this->type_ = g.type_;
    this->size_ = g.size_;
    this->qubit_id_ = vint();
    this->qubit_id_.assign(g.qubit_id_.begin(), g.qubit_id_.end());
    this->weight_ = g.weight_;
    this->name_ = g.name_;
}

Gate::~Gate() {}

int Gate::size() const { return this->size_; }

GateType Gate::type() const { return this->type_; }

vint Gate::qubit_id() const { return this->qubit_id_; }

Tensor<complex64> Gate::weight() const { return this->weight_; }

std::string Gate::name() const { return this->name_; }

void Gate::set_name(std::string name) { this->name_ = name; }
