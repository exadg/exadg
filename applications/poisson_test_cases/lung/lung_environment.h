#ifndef LUNG_ENVIRONMENT
#define LUNG_ENVIRONMENT

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include <deal.II/base/exceptions.h>

using namespace dealii;

void get_lung_files_from_environment(std::vector<std::string>& files) {

    if (const char* env_p = std::getenv("NAVIER_LUNG_FILES")) {
        char delimeter = '|';
        std::string s(env_p);
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delimeter))
            files.push_back(item);
    }
    
    AssertThrow(files.size() > 0, ExcMessage("The environment variable NAVIER_LUNG_FILES has not been set."));
    
}

#endif