#include "quantum/circuit.hpp"

#include "quantum/gate.hpp"
#include <cmath>
#include <iostream>

Circuit::Circuit(int qubit_number) { this->qubit_number_ = qubit_number; }

Circuit::~Circuit() {}

int Circuit::qubit_number() { return this->qubit_number_; }

Gate &Circuit::operator[](int index) { return this->gate[index]; }

void Circuit::add(Gate g) { this->gate.push_back(g); }

void Circuit::addI(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(0, 1) = 0;
    weight(1, 0) = 0;
    weight(1, 1) = 1;
    Gate g(GateType::kSingle, x, weight);
    g.set_name("I");
    this->gate.push_back(g);
}

void Circuit::addX(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 0;
    weight(0, 1) = 1;
    weight(1, 0) = 1;
    weight(1, 1) = 0;
    Gate g(GateType::kSingle, x, weight);
    g.set_name("X");
    this->gate.push_back(g);
}

void Circuit::addY(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 0;
    weight(0, 1) = complex64(0, -1);
    weight(1, 0) = complex64(0, 1);
    weight(1, 1) = 0;
    Gate g(GateType::kSingle, x, weight);
    g.set_name("Y");
    this->gate.push_back(g);
}

void Circuit::addZ(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(0, 1) = 0;
    weight(1, 0) = 0;
    weight(1, 1) = -1;
    Gate g(GateType::kSingle, x, weight);
    g.set_name("Z");
    this->gate.push_back(g);
}

void Circuit::addH(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1.0 / sqrt(2);
    weight(0, 1) = 1.0 / sqrt(2);
    weight(1, 0) = 1.0 / sqrt(2);
    weight(1, 1) = -1.0 / sqrt(2);
    Gate g(GateType::kSingle, x, weight);
    g.set_name("H");
    this->gate.push_back(g);
}

void Circuit::addU(int x, double theta, double lambda, double phi) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = cos(theta / 2);
    weight(0, 1) = -complex64(cos(lambda), sin(lambda)) * sin(theta / 2);
    weight(1, 0) = complex64(cos(phi), sin(phi)) * sin(theta / 2);
    weight(1, 1) = complex64(cos(lambda), sin(lambda)) *
                   complex64(cos(phi), sin(phi)) * cos(theta / 2);
    Gate g(GateType::kSingle, x, weight);
    g.set_name("U");
    this->gate.push_back(g);
}

void Circuit::addSqrtX(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1 / sqrt(2);
    weight(0, 1) = complex64(0, -1 / sqrt(2));
    weight(1, 0) = complex64(0, -1 / sqrt(2));
    weight(1, 1) = 1 / sqrt(2);
    Gate g(GateType::kSingle, x, weight);
    g.set_name("X^0.5");
    this->gate.push_back(g);
}

void Circuit::addSqrtY(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1 / sqrt(2);
    weight(0, 1) = -1 / sqrt(2);
    weight(1, 0) = 1 / sqrt(2);
    weight(1, 1) = 1 / sqrt(2);
    Gate g(GateType::kSingle, x, weight);
    g.set_name("Y^0.5");
    this->gate.push_back(g);
}

void Circuit::addSqrtW(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1 / sqrt(2);
    weight(0, 1) = -sqrt(complex64(0, 1)) / sqrt(2);
    weight(1, 0) = sqrt(complex64(0, -1)) / sqrt(2);
    weight(1, 1) = 1 / sqrt(2);
    Gate g(GateType::kSingle, x, weight);
    g.set_name("W^0.5");
    this->gate.push_back(g);
}

void Circuit::addT(int x) {
    Tensor<complex64> weight({2, 2}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(0, 1) = 0;
    weight(1, 0) = 0;
    weight(1, 1) = complex64(cos(M_PI / 4), sin(M_PI / 4));
    Gate g(GateType::kSingle, x, weight);
    g.set_name("T");
    this->gate.push_back(g);
}

void Circuit::addCNOT(int x, int y) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 1) = 1;
    weight(2, 3) = 1;
    weight(3, 2) = 1;
    Gate g(GateType::kControlled, x, y, weight);
    g.set_name("X");
    this->gate.push_back(g);
}

void Circuit::addCZ(int x, int y) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 1) = 1;
    weight(2, 2) = 1;
    weight(3, 3) = -1;
    Gate g(GateType::kControlled, x, y, weight);
    g.set_name("Z");
    this->gate.push_back(g);
}

void Circuit::addCP(int x, int y, double theta) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 1) = 1;
    weight(2, 2) = 1;
    weight(3, 3) = complex64(cos(theta), sin(theta));
    Gate g(GateType::kControlled, x, y, weight);
    g.set_name("P");
    this->gate.push_back(g);
}

void Circuit::addCU(int x, int y, double theta, double lambda, double phi) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 1) = 1;
    weight(2, 2) = cos(theta / 2);
    weight(2, 3) = -complex64(cos(lambda), sin(lambda)) * sin(theta / 2);
    weight(3, 2) = complex64(cos(phi), sin(phi)) * sin(theta / 2);
    weight(3, 3) = complex64(cos(lambda), sin(lambda)) *
                   complex64(cos(phi), sin(phi)) * cos(theta / 2);
    Gate g(GateType::kControlled, x, y, weight);
    g.set_name("U");
    this->gate.push_back(g);
}

void Circuit::addSwap(int x, int y) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 2) = 1;
    weight(2, 1) = 1;
    weight(3, 3) = 1;
    Gate g(GateType::kSwap, x, y, weight);
    this->gate.push_back(g);
}

void Circuit::addISwap(int x, int y, double theta) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 2) = complex64(0, -1);
    weight(2, 1) = complex64(0, -1);
    weight(3, 3) = complex64(cos(-theta), sin(-theta));
    Gate g(GateType::kDouble, x, y, weight);
    g.set_name("iSwap");
    this->gate.push_back(g);
}

void Circuit::addFSim(int x, int y, double theta, double phi) {
    Tensor<complex64> weight({4, 4}, true, Distribution::kGlobal);
    weight(0, 0) = 1;
    weight(1, 1) = cos(theta);
    weight(1, 2) = complex64(0, -sin(theta));
    weight(2, 1) = complex64(0, -sin(theta));
    weight(2, 2) = cos(theta);
    weight(3, 3) = complex64(cos(-phi), sin(-phi));
    Gate g(GateType::kDouble, x, y, weight);
    g.set_name("FSim");
    this->gate.push_back(g);
}

void Circuit::print() {
    std::string ret = "";
    for (int i = 0; i < this->qubit_number_; i++) {
        ret += "|0>--";
        for (Gate g : this->gate) {
            ret += "-";
            int len = g.name().length();
            switch (g.type()) {
            case GateType::kSingle: {
                if (g.qubit_id()[0] == i) {
                    ret += g.name();
                } else {
                    for (int j = 0; j < len; j++) {
                        ret += "-";
                    }
                }
                break;
            }

            case GateType::kSwap: {
                int x = g.qubit_id()[0];
                int y = g.qubit_id()[1];
                if (x == i || y == i) {
                    ret += "x";
                } else {
                    if (x < y && x < i && i < y) {
                        ret += "+";
                    } else if (x > y && x > i && i > y) {
                        ret += "+";
                    } else {
                        ret += "-";
                    }
                }
                break;
            }

            case GateType::kControlled: {
                int x = g.qubit_id()[0];
                int y = g.qubit_id()[1];
                int len = g.name().length();
                if (x == i) {
                    for (int j = 0; j < len / 2; j++) {
                        ret += "-";
                    }
                    ret += ".";
                    for (int j = 0; j < len - 1 - len / 2; j++) {
                        ret += "-";
                    }
                } else if (y == i) {
                    ret += g.name();
                } else {
                    if (x < y && x < i && i < y) {
                        for (int j = 0; j < len / 2; j++) {
                            ret += "-";
                        }
                        ret += "+";
                        for (int j = 0; j < len - 1 - len / 2; j++) {
                            ret += "-";
                        }
                    } else if (x > y && x > i && i > y) {
                        for (int j = 0; j < len / 2; j++) {
                            ret += "-";
                        }
                        ret += "+";
                        for (int j = 0; j < len - 1 - len / 2; j++) {
                            ret += "-";
                        }
                    } else {
                        for (int j = 0; j < len; j++) {
                            ret += "-";
                        }
                    }
                }
                break;
            }

            case GateType::kDouble: {
                int x = g.qubit_id()[0];
                int y = g.qubit_id()[1];
                int len = g.name().length();
                if (x == i || y == i) {
                    ret += g.name();
                } else {
                    if (x < y && x < i && i < y) {
                        for (int j = 0; j < len / 2; j++) {
                            ret += "-";
                        }
                        ret += "+";
                        for (int j = 0; j < len - 1 - len / 2; j++) {
                            ret += "-";
                        }
                    } else if (x > y && x > i && i > y) {
                        for (int j = 0; j < len / 2; j++) {
                            ret += "-";
                        }
                        ret += "+";
                        for (int j = 0; j < len - 1 - len / 2; j++) {
                            ret += "-";
                        }
                    } else {
                        for (int j = 0; j < len; j++) {
                            ret += "-";
                        }
                    }
                }
                break;
            }

            default:
                break;
            }
            ret += "-";
        }
        ret += "--";
        ret += "\n";
    }
    std::cerr << ret;
}