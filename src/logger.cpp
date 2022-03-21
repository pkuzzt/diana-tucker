#include "logger.hpp"

#include <chrono>
#include <stack>
#include <string>

std::stack<std::string> name_;
std::stack<std::chrono::steady_clock::time_point> time_;

void tic_() { t0 = std::chrono::steady_clock::now(); }