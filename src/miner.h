#ifndef PATTERN_MINER_H
#define PATTERN_MINER_H

#include <queue>

#include "pattern.h"
#include "typing.h"

class PatternMiner
{
public:
    PatternMiner(int alpha, int min_size, int max_size, double duration, int k, bool sort_alpha);
    std::vector<Pattern> mine(DiscreteDB &ts);
    Projection project(DiscreteDB &ts, Pattern pattern);

private:
    int alpha;
    int min_size;
    int max_size;
    double duration;
    int k;
    bool sort_alpha;
    int n;
    int max_duration;
    int max_gaps;
    std::priority_queue<Pattern, std::vector<Pattern>, std::greater<Pattern>> patterns;
    std::priority_queue<Pattern> queue;

    void mine_singletons(DiscreteDB &ts);
    Projection compute_projection_singleton(DiscreteDB &ts, int item);
    Projection compute_projection_incremental(DiscreteDB &ts, const Pattern &pattern, int item);
    Candidates get_candidates(DiscreteDB &ts, const Projection &projection, const std::vector<int> &pattern);
};

#endif
