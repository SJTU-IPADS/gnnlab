#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#define PAGE_SIZE 4096
class Dataset {
public:
    Dataset(std::string name, std::string folder) {
        auto tic = std::chrono::system_clock::now();

        key = name;

        if (folder.back() != '/') {
            folder.append("/");
        }

        // Load meta file
        std::string meta_filename = folder + "meta.json";
        std::ifstream meta_if(meta_filename);
        nlohmann::json j;
        meta_if >> j;

        this->num_nodes = j["num_nodes"];
        this->num_edges = j["num_edges"];
        this->num_train_ids = j["num_train_ids"];
        this->num_valid_ids = j["num_valid_ids"];
        this->num_test_ids = j["num_test_ids"];


        // Load data files
        std::string indptr_filename = folder + "indptr.bin";
        std::string indices_filename = folder + "indices.bin";
        std::string features_filename = folder + "features.bin";
        std::string labels_filename = folder + "labels.bin";
        std::string train_ids_filename = folder + "train_ids.bin";
        std::string valid_ids_filename = folder + "valid_ids.bin";
        std::string test_ids_filename = folder + "test_ids.bin";
        int indptr_fd = open(indptr_filename.c_str(), O_RDONLY, 0);
        int indices_fd = open(indices_filename.c_str(), O_RDONLY, 0);
        int features_fd = open(features_filename.c_str(), O_RDONLY, 0);
        int labels_fd = open(labels_filename.c_str(), O_RDONLY, 0);
        int train_ids_fd = open(train_ids_filename.c_str(), O_RDONLY, 0);
        int valid_ids_fd = open(valid_ids_filename.c_str(), O_RDONLY, 0);
        int test_ids_fd = open(test_ids_filename.c_str(), O_RDONLY, 0);

        indptr = (uint32_t *) mmap(NULL, (num_nodes + 1) * sizeof(uint32_t), PROT_READ, MAP_SHARED|MAP_FILE, indptr_fd, 0);
        indices = (uint32_t *) mmap(NULL, num_edges * sizeof(uint32_t), PROT_READ, MAP_SHARED|MAP_FILE, indices_fd, 0);
        features = (float *) mmap(NULL, num_nodes * feature_dim * sizeof(float), PROT_READ, MAP_SHARED|MAP_FILE, feature_dim, 0);
        labels = (uint32_t *) mmap(NULL, num_nodes * sizeof(uint32_t), PROT_READ, MAP_SHARED|MAP_FILE, labels_fd, 0);
        train_ids = (uint32_t *) mmap(NULL, num_train_ids * sizeof(uint32_t), PROT_READ, MAP_SHARED|MAP_FILE, train_ids_fd, 0);
        valid_ids = (uint32_t *) mmap(NULL, num_valid_ids * sizeof(uint32_t), PROT_READ, MAP_SHARED|MAP_FILE, valid_ids_fd, 0);
        test_ids = (uint32_t *) mmap(NULL, num_test_ids * sizeof(uint32_t), PROT_READ, MAP_SHARED|MAP_FILE, test_ids_fd, 0);

        assert(indptr);
        assert(indices);
        assert(features);
        assert(labels);
        assert(train_ids);
        assert(valid_ids);
        assert(test_ids);

        close(indptr_fd);
        close(indices_fd);
        close(features_fd);
        close(labels_fd);
        close(train_ids_fd);
        close(valid_ids_fd);
        close(test_ids_fd);
        
        // Make sure all data are loaded into memory
        for (uint32_t i = 0; i < (num_nodes + 1); i += PAGE_SIZE) {
            magic_num += indptr[i];
        }
        magic_num += indptr[num_nodes];

        for (uint32_t i = 0; i < num_edges; i += PAGE_SIZE) {
            magic_num += indices[i];
        }
        magic_num += indices[num_edges - 1];

        for (uint32_t i = 0; i < num_nodes * feature_dim; i += PAGE_SIZE) {
            magic_num += (uint32_t)features[i];
        }
        magic_num += (uint32_t)features[num_nodes * feature_dim - 1];

        for (uint32_t i = 0; i < num_nodes; i += PAGE_SIZE) {
            magic_num += labels[i];
        }
        magic_num += labels[num_nodes - 1];

        for (uint32_t i = 0; i < num_train_ids; i += PAGE_SIZE) {
            magic_num += train_ids[i];
        }
        magic_num += train_ids[num_train_ids - 1];

        for (uint32_t i = 0; i < num_valid_ids; i += PAGE_SIZE) {
            magic_num += valid_ids[i];
        }
        magic_num += valid_ids[num_valid_ids - 1];

        for (uint32_t i = 0; i < num_test_ids; i += PAGE_SIZE) {
            magic_num += test_ids[i];
        }
        magic_num += test_ids[num_test_ids - 1];

        auto toc = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = toc - tic;

        printf("Loading %s dataset using %.4f secs.", key.c_str(), duration.count());
    }

    ~Dataset() {
        munmap(indptr, (num_nodes + 1) * sizeof(uint32_t));
        munmap(indices, num_edges * sizeof(uint32_t));
        munmap(features, num_nodes * feature_dim * sizeof(float));
        munmap(labels, num_nodes * sizeof(uint32_t));
        munmap(train_ids, num_train_ids * sizeof(uint32_t));
        munmap(valid_ids, num_valid_ids * sizeof(uint32_t));
        munmap(test_ids, num_test_ids * sizeof(uint32_t));
    }

private:
    std::string key;
    uint32_t *indptr = nullptr;
    uint32_t *indices = nullptr;
    
    float *features = nullptr;
    uint32_t *labels = nullptr;

    uint32_t *train_ids = nullptr;
    uint32_t *valid_ids = nullptr;
    uint32_t *test_ids = nullptr;

    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t feature_dim;
    uint32_t num_train_ids;
    uint32_t num_valid_ids;
    uint32_t num_test_ids;

    uint32_t magic_num = 0; 
};
