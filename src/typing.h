#ifndef TYPING_H
#define TYPING_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

using Word = std::vector<int>;
using DiscreteDB = const std::vector<Word>;
using Projection = std::unordered_map<int, std::pair<int, int>>;
using Candidates = std::unordered_set<int>;
#endif
