#include "quantum/qregister.hpp"

QRegister::QRegister(int qubit_number) { this->qubit_number_ = qubit_number; }

QRegister::~QRegister() {}

int QRegister::qubit_number() { return this->qubit_number_; }