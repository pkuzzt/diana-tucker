#ifndef __DIANA_CORE_INCLUDE_DEBUGGER_HPP__
#define __DIANA_CORE_INCLUDE_DEBUGGER_HPP__

// LOG_LEVEL 0 debug, tick
// LOG_LEVEL 1 info
// LOG_LEVEL 2 warn, checkwarn
// LOG_LEVEL 3 error, checkerr
// LOG_LEVEL 4 fatal, assert

#ifndef LOG_LEVEL
#define LOG_LEVEL 0
#endif

#define COLOR_RESET "\033[0m"
#define COLOR_DEBUG "\033[36m" /* Cyan */
#define COLOR_INFO "\033[32m"  /* Green */
#define COLOR_WARN "\033[33m"  /* Yellow */
#define COLOR_ERROR "\033[35m" /* Magenta */
#define COLOR_FATAL "\033[31m" /* Red */

#include <iostream>
#include <string>

#define debug(x)                                                               \
    do {                                                                       \
        if (LOG_LEVEL <= 0) {                                                  \
            std::cerr << "[" COLOR_DEBUG "DEBUG" COLOR_RESET " "               \
                      << __FILE__ << ":" << __LINE__ << "] " << #x             \
                      << " = " << (x) << std::endl;                            \
        }                                                                      \
    } while (0)

#define tick                                                                   \
    do {                                                                       \
        if (LOG_LEVEL <= 0) {                                                  \
            std::cerr << "[" COLOR_DEBUG "TICK" COLOR_RESET " "                \
                      << __FILE__ << ":" << __LINE__ << "] " << std::endl;     \
        }                                                                      \
    } while (0)

#define info(x)                                                                \
    do {                                                                       \
        if (LOG_LEVEL <= 1) {                                                  \
            std::cerr << "[" COLOR_INFO "INFO" COLOR_RESET " " << __FILE__     \
                      << ":" << __LINE__ << "] " << (x) << std::endl;          \
        }                                                                      \
    } while (0)

#define output(x)                                                              \
    do {                                                                       \
        if (mpi_rank() == 0) {                                                 \
            std::cerr << (x) << std::endl;                                     \
        }                                                                      \
    } while (0)

#define warn(x)                                                                \
    do {                                                                       \
        if (LOG_LEVEL <= 2) {                                                  \
            std::cerr << "[" COLOR_WARN "WARN" COLOR_RESET " " << __FILE__     \
                      << ":" << __LINE__ << "] " << (x) << std::endl;          \
        }                                                                      \
    } while (0)

#define checkwarn(x)                                                           \
    do {                                                                       \
        if (LOG_LEVEL <= 2) {                                                  \
            if (!(x)) {                                                        \
                std::cerr << "[" COLOR_WARN "WARN" COLOR_RESET " "             \
                          << __FILE__ << ":" << __LINE__ << "] " << #x         \
                          << std::endl;                                        \
            }                                                                  \
        }                                                                      \
    } while (0)

#define error(x)                                                               \
    do {                                                                       \
        if (LOG_LEVEL <= 3) {                                                  \
            std::cerr << "[" COLOR_ERROR "ERROR" COLOR_RESET " "               \
                      << __FILE__ << ":" << __LINE__ << "] " << (x)            \
                      << std::endl;                                            \
            exit(-3);                                                          \
            __builtin_unreachable();                                           \
        }                                                                      \
    } while (0)

#undef checkerr
#define checkerr(x)                                                            \
    do {                                                                       \
        if (LOG_LEVEL <= 3) {                                                  \
            if (!(x)) {                                                        \
                std::cerr << "[" COLOR_ERROR "ERROR" COLOR_RESET " "           \
                          << __FILE__ << ":" << __LINE__ << "] " << #x         \
                          << std::endl;                                        \
                exit(-3);                                                      \
                __builtin_unreachable();                                       \
            }                                                                  \
        }                                                                      \
    } while (0)

#define fatal(x)                                                               \
    do {                                                                       \
        if (LOG_LEVEL <= 4) {                                                  \
            std::cerr << "[" COLOR_FATAL "FATAL" COLOR_RESET " "               \
                      << __FILE__ << ":" << __LINE__ << "] " << (x)            \
                      << std::endl;                                            \
            exit(-4);                                                          \
            __builtin_unreachable();                                           \
        }                                                                      \
    } while (0)

#undef assert
#define assert(x)                                                              \
    do {                                                                       \
        if (LOG_LEVEL <= 4) {                                                  \
            if (!(x)) {                                                        \
                std::cerr << "[" COLOR_FATAL "ASSERTION FAIL" COLOR_RESET " "  \
                          << __FILE__ << ":" << __LINE__ << "] " << #x         \
                          << std::endl;                                        \
                exit(-4);                                                      \
                __builtin_unreachable();                                       \
            }                                                                  \
        }                                                                      \
    } while (0)

#define pbp(x)                                                                 \
    do {                                                                       \
        auto comm = new Communicator<int>();                                   \
        for (int i = 0; i < comm->size(); i++) {                               \
            comm->barrier();                                                   \
            if (i == comm->rank()) {                                           \
                info("Rank " + std::to_string(i) + ":");                       \
                { x; }                                                         \
            }                                                                  \
        }                                                                      \
        delete comm;                                                           \
    } while (0)

#define print_vec(x)                                                           \
    do {                                                                       \
        for (auto item : x) {                                                  \
            std::cerr << item << " ";                                          \
        }                                                                      \
        std::cerr << std::endl;                                                \
    } while (0)

void tic_();

void toc_();

#define tic                                                                    \
    do {                                                                       \
        tic_();                                                                \
    } while (0)

#define toc                                                                    \
    do {                                                                       \
        toc_();                                                                \
    } while (0)

#endif