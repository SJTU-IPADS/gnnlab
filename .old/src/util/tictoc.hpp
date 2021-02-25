#pragma once

#include <chrono>
#include <unordered_map>

class TicToc {
public:
    inline void Tic(int tag) {
        tics[tag] = std::chrono::system_clock::now();
    }

    inline double Toc(int tag) {
        std::chrono::system_clock::time_point toc = std::chrono::system_clock::now();
        auto it = tics.find(tag);
        if (it == tics.end()) {
            return -1.0f;
        }
        std::chrono::duration<double> duration = toc - it->second;
        return duration.count();
    }

private:
    std::unordered_map<int, std::chrono::system_clock::time_point> tics;
};
