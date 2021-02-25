#pragma once

#include <sys/stat.h>

#include <string>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "data/dataset.hpp"

enum DataStoreErrCode{
    kSuccess = 0,
    kDuplicateKey,
    kAlreadyLoad,
    kFolderNotFound
};

struct DataStoreErr{
    DataStoreErrCode code;
    std::string message;
};

class DataStore {
public:
    DataStoreErr Load(std::string key, std::string folder) {

        struct stat st;
        if (stat (folder.c_str(), &st) != 0) {
            return {kFolderNotFound, "folder not exists"};
        }

        auto folder_it = folder2key.find(folder);
        if (folder_it != folder2key.end()) {
            return {kAlreadyLoad, "data has been loaded with key " + folder_it->second};
        }

        auto key_it = store.find(key);
        if (key_it != store.end()) {
            return {kDuplicateKey, "duplicate key"};
        }

        store[key] = std::make_shared<Dataset>(key, folder);
        folder2key[folder] = key;

        return {kSuccess, "success"};
    }

    void Remove(std::string key) {
        auto store_it = store.find(key);
        if (store_it != store.end()) {
            std::string folder = store_it->second->folder;
            store.erase(store_it);
            folder2key.erase(folder);
        }
    }

    void List() {
        for (auto it : store) {
            std::cout << it.first << " : " << it.second->folder << std::endl;
        }
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Dataset>> store;
    std::unordered_map<std::string, std::string> folder2key;
};
