//
// Created by 丁明朔 on 2022/3/3.
//

#ifndef DIANA_TUCKER_SUMMARY_HPP
#define DIANA_TUCKER_SUMMARY_HPP

#include <string>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <map>

#include "def.hpp"

inline std::string method_name(const std::string &prettyFunction) {
    size_t colons = prettyFunction.find("::");
    size_t begin = prettyFunction.substr(0, colons).rfind(' ') + 1;
    size_t end = prettyFunction.rfind('(') - begin;

    return prettyFunction.substr(begin, end) + "()";
}

#define  METHOD_NAME method_name(__PRETTY_FUNCTION__)


class Summary {
private:
    struct Event {
        std::string name;

        size_t caller_id;

        long long flop;
        long long bandwidth;

        double time_start;
        double time_end;
        double time_length;
        double time_counted;

        std::vector<size_t> callee_ids;
    };
    static std::vector<Event> events_;
    static std::map<std::string, std::vector<size_t>> events_name_map_;
    static size_t last_id_;
    static bool recording_;

public:
    static void init();

    static void finalize();

    static void start(const std::string &name, long long flop = 0,
                      long long bandwidth = 0);

    static void end(const std::string &name);


    static void print_summary();
};

#endif //DIANA_TUCKER_SUMMARY_HPP
