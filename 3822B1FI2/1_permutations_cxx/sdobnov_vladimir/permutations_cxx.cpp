#include "permutations_cxx.h"

#include <unordered_map>
#include <array>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
	std::unordered_map<std::string, std::vector<std::string>> groups;
	std::unordered_map<std::string, std::string> cache;

	for (const auto& [word, _] : dictionary) {
		auto key = make_key(word);
		cache[word] = key;
		groups[key].push_back(word);
	}

	for (auto& [word, vec] : dictionary) {
		const auto& key = cache[word];
		const auto& group = groups[key];

		vec.reserve(group.size() > 0 ? group.size() - 1 : 0);

		for (const auto& other : group) {
			if (other != word) {
				vec.push_back(other);
			}
		}

		std::sort(vec.rbegin(), vec.rend());
	}
}