#ifndef __DIANA_CORE_INCLUDE_QUANTUM_CIRCUIT_HPP__
#define __DIANA_CORE_INCLUDE_QUANTUM_CIRCUIT_HPP__

#include <vector>

#include "quantum/gate.hpp"

class Circuit {
private:
    int qubit_number_;

public:
    std::vector<Gate> gate;

    Circuit(int);

    ~Circuit();

    int qubit_number();

    Gate &operator[](int index);

    void add(Gate g);

    void addI(int x);

    void addX(int x);

    void addY(int x);

    void addZ(int x);

    void addH(int x);

    void addSqrtX(int x);

    void addSqrtY(int x);

    void addSqrtW(int x);

    void addT(int x);

    void addU(int x, double theta, double lambda, double phi);

    void addCNOT(int x, int y);

    void addCZ(int x, int y);

    void addCP(int x, int y, double theta);

    void addCU(int x, int y, double theta, double lambda, double phi);

    void addSwap(int x, int y);

    void addISwap(int x, int y, double theta);

    void addFSim(int x, int y, double theta, double phi);

    void print();
};

#endif
