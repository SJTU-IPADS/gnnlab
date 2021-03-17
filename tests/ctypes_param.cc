#include <cstdint>
#include <cstdio>

extern "C" uint64_t big_num(uint64_t i) {
    printf("big num got input %llu\n", i);
    return i;
}