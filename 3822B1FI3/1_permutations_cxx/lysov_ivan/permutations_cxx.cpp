#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    groups.reserve(dictionary.size());

    for (const auto& [word, _] : dictionary) {
        std::string signature = word;
        std::sort(signature.begin(), signature.end());
        groups[signature].push_back(word);
    }

    for (auto& [word, permutations] : dictionary) {
        std::string signature = word;
        std::sort(signature.begin(), signature.end());

        const auto& group = groups[signature];
        permutations.clear();
        permutations.reserve(group.size() > 0 ? group.size() - 1 : 0);

        for (const auto& candidate : group) {
            if (candidate != word) {
                permutations.push_back(candidate);
            }
        }

        std::sort(permutations.begin(), permutations.end(), std::greater<std::string>());
    }
}
