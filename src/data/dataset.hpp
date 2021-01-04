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
#include <iostream>

#include <nlohmann/json.hpp>

#define PAGE_SIZE 4096
class Dataset {
public:
    const std::string key;
    const std::string folder;

    Dataset(std::string name, std::string folder, bool lock=true) : key(name), folder(folder) {
        auto tic = std::chrono::system_clock::now();

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
        this->feature_dim = j["feat_dim"];
        this->num_classes = j["num_classes"];
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
        features = (float *) mmap(NULL, num_nodes * feature_dim * sizeof(float), PROT_READ, MAP_SHARED|MAP_FILE, features_fd, 0);
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

        // Make sure all data are loaded into memory
        if (lock) {
            mlock(indptr, (num_nodes + 1) * sizeof(uint32_t));
            mlock(indices, num_edges * sizeof(uint32_t));
            mlock(features, num_nodes * feature_dim * sizeof(float));
            mlock(labels, num_nodes * sizeof(uint32_t));
            mlock(train_ids, num_train_ids * sizeof(uint32_t));
            mlock(valid_ids, num_valid_ids * sizeof(uint32_t));
            mlock(test_ids, num_test_ids * sizeof(uint32_t));
        }

        auto toc = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = toc - tic;

        printf("Loaded %s dataset with num_nodes: %lu, num_edges: %lu, feat_dims: %lu, num_classes: %lu using %.4f secs.\n", 
               key.c_str(), num_nodes, num_edges, feature_dim, num_classes, duration.count());

        close(indptr_fd);
        close(indices_fd);
        close(features_fd);
        close(labels_fd);
        close(train_ids_fd);
        close(valid_ids_fd);
        close(test_ids_fd);
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
    uint32_t *indptr = nullptr;
    uint32_t *indices = nullptr;
    
    float *features = nullptr;
    uint32_t *labels = nullptr;

    uint32_t *train_ids = nullptr;
    uint32_t *valid_ids = nullptr;
    uint32_t *test_ids = nullptr;

    size_t num_nodes;
    size_t num_edges;
    size_t feature_dim;
    size_t num_classes;
    size_t num_train_ids;
    size_t num_valid_ids;
    size_t num_test_ids;
};
