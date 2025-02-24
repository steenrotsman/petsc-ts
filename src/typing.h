#ifndef TYPING_H
#define TYPING_H

#include <vector>
#include <unordered_map>
#include <unordered_set>

using DiscreteDB = const std::vector<std::vector<int>>;
using Projection = std::unordered_map<int, std::pair<int, int>>;
using Candidates = std::unordered_set<int>;

#endif