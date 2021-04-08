#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <string>
#include <atomic>
#include <vector>
#include <iostream>

int main() {
    std::vector<std::atomic<int>> degree(10000000);
    std::string filepath = "/graph-learning/samgraph/papers100M/indptr.bin";
    struct stat st;
    stat(filepath.c_str(), &st);
    size_t size = st.st_size;

    int fd = open(filepath.c_str(), O_RDONLY, 0);
    int *data = (int*)mmap(NULL, size, PROT_READ, MAP_SHARED|MAP_FILE, fd, 0);
    
    #pragma omp parallel for
    for (size_t i = 0; i < 111059956; i++) {
        size_t len = data[i + 1] - data[i];
        std::atomic_fetch_add(&(degree.at(len)), 1);
    }

    for (size_t i = 0; i < degree.size(); i++) {
        if (degree[i] != 0) {
            std::cout << i << " " << degree[i] << std::endl;
        }
    }
    close(fd);
}
