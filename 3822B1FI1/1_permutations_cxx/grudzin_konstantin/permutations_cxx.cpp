#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dict) {
    std::unordered_map<std::string, std::vector<std::string>> gr;

    for (auto const& item : dict) {
        const std::string& key = item.first;
        std::string sorted_key = key;
        std::sort(sorted_key.begin(), sorted_key.end());
        
        gr[sorted_key].push_back(key);
    }

    for (auto& item : dict) {
        const std::string& key = item.first;
        std::vector<std::string>& value = item.second;

        std::string sorted_key = key;
        std::sort(sorted_key.begin(), sorted_key.end());

        const auto& candidates = gr[sorted_key];

        for (const auto& variant : candidates) {
            if (variant != key) {
                value.push_back(variant);
            }
        }

        std::sort(value.rbegin(), value.rend());
    }
}