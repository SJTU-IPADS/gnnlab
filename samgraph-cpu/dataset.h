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
#include <unordered_map> 
#include <sstream>
#include <iterator>

#include "graph_storage.h"

#define PAGE_SIZE 4096

struct Feature {
    const float *data;
    const size_t dim;

    Feature(float *feature, size_t dim) : data(feature), dim(dim) {}
};

struct Label {
    const uint32_t *data;
    const size_t num_classes;

    Label(uint32_t *label, size_t num_classes) : data(label), num_classes(num_classes) {}
};

struct NodeSet {
    const uint32_t *ids;
    const size_t len;

    NodeSet(uint32_t *ids, size_t len) : ids(ids), len(len) {}
};

class Dataset {
public:
    const std::string key;
    const std::string folder;

    Dataset(std::string name, std::string folder) : key(name), folder(folder) {
        auto tic = std::chrono::system_clock::now();

        if (folder.back() != '/') {
            folder.append("/");
        }

        std::unordered_map<std::string, size_t> meta;

        // Load meta file
        std::string meta_filename = folder + "meta.txt";
        std::ifstream meta_file(meta_filename);
        std::string line;
        while(std::getline(meta_file, line)) {
            std::istringstream iss(line);
            std::vector<std::string> kv {std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

            if (kv.size() < 2) {
                break;
            }

            meta[kv[0]] = std::stoull(kv[1]);
        }

        this->num_nodes = meta["NUM_NODE"];
        this->num_edges = meta["NUM_EDGE"];
        this->feature_dim = meta["FEAT_DIM"];
        this->num_classes = meta["NUM_CLASS"];
        this->num_train_ids = meta["NUM_TRAIN_SET"];
        this->num_valid_ids = meta["NUM_VALID_SET"];
        this->num_test_ids = meta["NUM_TEST_SET"];

        // Load data files
        std::string indptr_filename = folder + "indptr.bin";
        std::string indices_filename = folder + "indices.bin";
        std::string features_filename = folder + "feat.bin";
        std::string labels_filename = folder + "label.bin";
        std::string train_ids_filename = folder + "train_set.bin";
        std::string valid_ids_filename = folder + "valid_set.bin";
        std::string test_ids_filename = folder + "test_set.bin";
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
        mlock(indptr, (num_nodes + 1) * sizeof(uint32_t));
        mlock(indices, num_edges * sizeof(uint32_t));
        mlock(features, num_nodes * feature_dim * sizeof(float));
        mlock(labels, num_nodes * sizeof(uint32_t));
        mlock(train_ids, num_train_ids * sizeof(uint32_t));
        mlock(valid_ids, num_valid_ids * sizeof(uint32_t));
        mlock(test_ids, num_test_ids * sizeof(uint32_t));

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

    std::shared_ptr<CSR> GetCSR() {
        std::shared_ptr<CSR> csr = std::make_shared<CSR>();
        csr->num_rows = num_nodes;
        csr->num_cols = num_nodes;
        csr->num_edges = num_edges;
        csr->indptr = indptr;
        csr->indices = indices;
        csr->sorted = true;
        csr->need_free = false;

        return csr;
    }

    inline Feature GetFeature() {
        return {features, feature_dim};
    }

    inline Label GetLabel() {
        return {labels, num_classes};
    }

    inline NodeSet GetTrainSet() {
        return {train_ids, num_train_ids};
    }

    inline NodeSet GetTestSet() {
        return {test_ids, num_test_ids};
    }

    inline NodeSet GetValidSet() {
        return {valid_ids, num_valid_ids};
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
