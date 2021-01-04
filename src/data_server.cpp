#include <signal.h>

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <sstream>
#include <algorithm>

#include "data/data_store.hpp"

void my_handler(int s){
    printf("\nCaught signal %d\n", s);
    exit(1);
}

void register_handler() {
    signal(SIGINT, my_handler);
}
std::vector<std::string> split_string(std::string str) {
    std::istringstream iss(str);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
    return tokens;
}
int main() {
    register_handler();

    DataStore datastore;

    std::string command;
    while(true) {
        std::cout << "prompt> ";
        std::getline(std::cin, command);
        if (command.find("load") == 0) {
            std::vector<std::string> tokens = split_string(command);
            if (tokens.size() < 3) {
                std::cout << "usage: load key /path/to/folder" << std::endl;
                continue;
            }
            DataStoreErr err = datastore.Load(tokens[1], tokens[2]);
            std::cout << err.message << std::endl;
        } else if (command.find("remove") == 0) {
            std::vector<std::string> tokens = split_string(command);
            if (tokens.size() < 3) {
                std::cout << "usage: load key /path/to/folder" << std::endl;
                continue;
            }
            datastore.Remove(tokens[1]);
        } else if (command.find("list") == 0) {
            datastore.List();
        } else if (command.find("echo") == 0) {
            std::cout << command << std::endl;
        } else if (command.find("exit") == 0) {
            break;
        } else {
            std::cout << "unknown command" << std::endl;
        }
    }
}
