#include <vector>
#include <queue>
#include <algorithm>

#include "miner.h"
#include "pattern.h"
#include "typing.h"

PatternMiner::PatternMiner(int alpha, int min_size, int max_size, double duration, int k, bool sort_alpha) : alpha(alpha), min_size(min_size), max_size(max_size), duration(duration), k(k), sort_alpha(sort_alpha), n(0)
{
    max_duration = max_size * duration;
    max_gaps = max_duration - max_size;
}

std::vector<Pattern> PatternMiner::mine(DiscreteDB &ts)
{
    // Reset from previous calls
    n = 0;
    patterns = std::priority_queue<Pattern, std::vector<Pattern>, std::greater<Pattern>>();
    queue = std::priority_queue<Pattern>();

    // Initializes queue with singletons
    mine_singletons(ts);

    // Levels n + 1
    while (!(queue.empty() || (n == k && queue.top() < patterns.top())))
    {
        Pattern pattern = queue.top();
        queue.pop();

        // Only generate candidates if they are short enough
        if (pattern.pattern.size() < static_cast<size_t>(max_size))
        {
            for (int item : pattern.candidates)
            {
                std::vector<int> candidate = pattern.pattern;
                candidate.push_back(item);
                auto projection = compute_projection_incremental(ts, pattern, item);

                // Don't add candidate to queue if its support is lower than kth patttern
                if (n < k || static_cast<int>(projection.size()) > patterns.top().support)
                {
                    auto candidates = get_candidates(ts, projection, candidate);
                    queue.emplace(candidate, projection, candidates);
                }
            }
        }
        
        // Add pattern to patterns if length and support are high enough
        if (pattern.pattern.size() >= static_cast<size_t>(min_size))
        {
            if (n < k)
            {
                patterns.push(pattern);
                ++n;
            } else if (pattern > patterns.top())
            {
                patterns.pop();
                patterns.push(pattern);
            }
        }
    }

    // Turn heap patterns into sorted vector
    std::vector<Pattern> result;
    result.reserve(patterns.size());

    while (!patterns.empty())
    {
        result.push_back(patterns.top());
        patterns.pop();
    }
    if (sort_alpha)
    {
        std::sort(result.begin(), result.end(), [](const Pattern &a, const Pattern &b)
        {
            return a.pattern < b.pattern;
        });
    }
    return result;
}

void PatternMiner::mine_singletons(DiscreteDB &ts)
{
    // Assume that every sax symbol occurs at least once
    for (int item = 0; item < alpha; ++item)
    {
        std::vector<int> pattern = {item};
        auto projection = compute_projection_singleton(ts, item);
        auto candidates = get_candidates(ts, projection, pattern);
        queue.emplace(pattern, projection, candidates);
    }
}

Projection PatternMiner::compute_projection_singleton(DiscreteDB &ts, int item)
{
    Projection projection;
    for (size_t ts_idx = 0; ts_idx < ts.size(); ++ts_idx)
    {
        for (size_t j = 0; j + min_size <= ts[ts_idx].size(); ++j)
        {
            if (ts[ts_idx][j] == item)
            {
                projection[ts_idx] = {static_cast<int>(j), static_cast<int>(j + 1)};
                break;
            }
        }
    }
    return projection;
}

Projection PatternMiner::compute_projection_incremental(DiscreteDB &ts, const Pattern &pattern, int item)
{
    Projection p;
    for (const auto &[ts_idx, range] : pattern.projection)
    {
        int start = range.first, end = range.second;
        int current_max_size = start + max_duration;
        int current_gaps = (end - start) - pattern.pattern.size();
        int remaining_gaps = max_gaps - current_gaps;
        int stop = std::min({static_cast<int>(ts[ts_idx].size()), current_max_size, end + remaining_gaps + 1});

        for (int j = end; j < stop; ++j)
        {
            if (ts[ts_idx][j] == item)
            {
                p[ts_idx] = {start, j + 1};
                break;
            }
        }
    }
    return p;
}

Candidates PatternMiner::get_candidates(DiscreteDB &ts, const Projection &projection, const std::vector<int> &pattern)
{
    Candidates candidates;
    for (const auto &[ts_idx, range] : projection)
    {
        if (candidates.size() == static_cast<size_t>(alpha)) break;

        int start = range.first, end = range.second;
        int current_max_size = start + max_duration;
        int current_gaps = (end - start) - pattern.size();
        int remaining_gaps = max_gaps - current_gaps;
        int stop = std::min({static_cast<int>(ts[ts_idx].size()), current_max_size, end + remaining_gaps + 1});

        for (int j = end; j < stop; ++j)
        {
            candidates.insert(ts[ts_idx][j]);
        }
    }
    return candidates;
}
