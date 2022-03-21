//
// Created by 丁明朔 on 2022/3/3.
//

#include <queue>
#include <iostream>

#include "summary.hpp"
#include "logger.hpp"
#include "communicator.hpp"

#define ROOT_ID SIZE_MAX
std::vector<Summary::Event> Summary::events_ = std::vector<Summary::Event>();
std::map<std::string, std::vector<size_t>>  Summary::events_name_map_ =
        std::map<std::string, std::vector<size_t>>();
size_t Summary::last_id_ = ROOT_ID;
bool Summary::recording_ = false;

void Summary::init() {
    Summary::recording_ = true;
    Summary::start("{Main}");
}

void Summary::finalize() {
    Summary::end("{Main}");
    Summary::recording_ = false;
}

void
Summary::start(const std::string &name, long long flop,
               long long bandwidth) {
    if (!Summary::recording_) {
        return;
    }
    Summary::events_.push_back({
                                       name,
                                       Summary::last_id_,
                                       flop,
                                       bandwidth,
                                       MPI_Wtime(),
                                       0,
                                       0,
                                       0,
                                       std::vector<size_t>()
                               });
    Summary::last_id_ = events_.size() - 1;
    Summary::events_name_map_[name].push_back(Summary::last_id_);
}

void
Summary::end(const std::string &name) {
    if (!Summary::recording_) {
        return;
    }
    size_t idx = Summary::last_id_;
    Summary::Event &event = Summary::events_[idx];
    assert(event.name == name);
    event.time_end = MPI_Wtime();
    event.time_length = event.time_end - event.time_start;
    Summary::last_id_ = event.caller_id;
    if (Summary::last_id_ != ROOT_ID) {
        // If last_id_ is not root, then gather the data to the caller.
        Summary::Event &caller_event = Summary::events_[Summary::last_id_];
        caller_event.callee_ids.push_back(idx);
        caller_event.flop += event.flop;
        caller_event.bandwidth += event.bandwidth;
        caller_event.time_counted += event.time_length;
    }
}

void fill_space_(std::string &s, size_t len) {
    while (s.length() < len) {
        s += " ";
    }
}

void add_data_(std::string &output, const std::string &data, size_t len) {
    std::string tmp = data;
    fill_space_(tmp, len);
    output += tmp;
}

void add_separate_line_(std::string &output, int first, int second, int third) {
    output += "|-";
    for (int i = 0; i < first; i++) {
        output += "-";
    }
    output += "+-";
    for (int i = 0; i < second; i++) {
        output += "-";
    }
    output += "+-";
    for (int i = 0; i < third; i++) {
        output += "-";
    }
    output += "| ";
    output += "\n";
}

void Summary::print_summary() {
    const int kCaptionLength = 15;
    const int kFirstSectionLength = 10;
    const std::string kSeparate = "| ";
    const std::string kSecondSectionCaption[] = {"Time(s)", "Time C.(%)",
                                                 "Number",
                                                 "Avg. Time(s)"};
    const std::string kThirdSectionCaption[] = {"GFlop/s", "Bandw.(GB/s)"};
    std::string output;
    // Display caption row.
    add_separate_line_(output, kFirstSectionLength, 4 * kCaptionLength,
                       2 * kCaptionLength);
    output += kSeparate;
    add_data_(output, "", kFirstSectionLength);
    output += kSeparate;
    for (const std::string &caption: kSecondSectionCaption) {
        add_data_(output, caption, kCaptionLength);
    }
    output += kSeparate;
    for (const std::string &caption: kThirdSectionCaption) {
        add_data_(output, caption, kCaptionLength);
    }
    output += kSeparate;
    output += "\n";
    add_separate_line_(output, kFirstSectionLength, 4 * kCaptionLength,
                       2 * kCaptionLength);
    // Display events.
    for (const auto &event_list: Summary::events_name_map_) {
        // Get important statistics.
        double time_length_total = 0;
        double time_length_counted = 0;
        long long flop = 0;
        long long bandwidth = 0;
        long long flop_global = 0;
        long long bandwidth_global = 0;
        size_t number = event_list.second.size();
        for (const auto &idx: event_list.second) {
            const auto &event = Summary::events_[idx];
            time_length_total += event.time_length;
            time_length_counted += event.time_counted;
            flop += event.flop;
            bandwidth += event.bandwidth;
        }
        // First section,  and first line, contains name and global data.
        output += kSeparate;
        add_data_(output, event_list.first,
                  kFirstSectionLength + 4 * kCaptionLength +
                  kSeparate.length());
        output += kSeparate;
        if (event_list.first == "{Main}") {
            (new Communicator<long long>)->allreduce(&flop,
                                                     &flop_global, 1,
                                                     MPI_SUM);
            (new Communicator<long long>)->allreduce(&bandwidth,
                                                     &bandwidth_global,
                                                     1, MPI_SUM);
            add_data_(output,
                      std::to_string((double) flop_global / 1e9 /
                                     time_length_total),
                      kCaptionLength);
            add_data_(output,
                      std::to_string(
                              (double) bandwidth_global / 1073741824 /
                              time_length_total),
                      kCaptionLength);
        } else {
            add_data_(output, "", 2 * kCaptionLength);
        }
        output += kSeparate;
        output += "\n";
        // Second line
        output += kSeparate;
        add_data_(output, "", kFirstSectionLength);
        output += kSeparate;
        // Second section, contains "Time", "Time Counted", "Time Counted%",
        // "Number", "Average Time".
        add_data_(output, std::to_string(time_length_total),
                  kCaptionLength);
        if (time_length_counted == 0) {
            add_data_(output, "Kernel", kCaptionLength);
        } else {
            add_data_(output,
                      std::to_string(
                              time_length_counted / time_length_total * 100),
                      kCaptionLength);
        }
        add_data_(output, std::to_string(number), kCaptionLength);
        add_data_(output, std::to_string(time_length_total / (double) number),
                  kCaptionLength);
        output += kSeparate;
        // Third section, contains flop/s, bandwidth/s.
        add_data_(output,
                  std::to_string((double) flop / 1e9 / time_length_total),
                  kCaptionLength);
        add_data_(output,
                  std::to_string(
                          (double) bandwidth / 1073741824 / time_length_total),
                  kCaptionLength);
        output += kSeparate;
        output += "\n";
        add_separate_line_(output, kFirstSectionLength, 4 * kCaptionLength,
                           2 * kCaptionLength);
    }
    if (mpi_rank() == 0) {
        std::cerr << output << std::endl;
    }
}

