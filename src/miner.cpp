#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

#include "miner.h"
#include "pattern.h"
#include "typing.h"

// --- Constructor ---
PatternMiner::PatternMiner(int alpha, int min_size, int max_size,
                           double duration, int k, bool sort_alpha)
    : alpha(alpha), min_size(min_size), max_size(max_size), duration(duration),
      k(k), sort_alpha(sort_alpha), n(0) {
  max_duration = max_size * duration;
  max_gaps = max_duration - max_size;
}

// --- Public Methods ---
std::vector<Pattern> PatternMiner::mine(DiscreteDB &ts) {
  // Reset from previous calls
  n = 0;
  patterns = TopKPatterns();
  queue = PatternQueue();

  // Initializes queue with singletons
  mine_singletons(ts);

  // Levels n + 1
  while (!(queue.empty() || (n == k && queue.top() < patterns.top()))) {
    Pattern pattern = queue.top();
    queue.pop();

    // Only generate candidates if they are short enough
    if (pattern.pattern.size() < static_cast<size_t>(max_size)) {
      for (int item : pattern.candidates) {
        Word candidate = pattern.pattern;
        candidate.push_back(item);
        auto [projection, support] = project_incremental(ts, pattern, item);

        // Don't add candidate to queue if its support is lower than kth
        // patttern
        if (n < k || support > patterns.top().support) {
          auto candidates = get_candidates(ts, projection, candidate);
          queue.emplace(candidate, projection, candidates, support);
        }
      }
    }

    // Add pattern to patterns if length and support are high enough
    if (pattern.pattern.size() >= static_cast<size_t>(min_size)) {
      if (n < k) {
        patterns.push(pattern);
        ++n;
      } else if (pattern > patterns.top()) {
        patterns.pop();
        patterns.push(pattern);
      }
    }
  }

  // Turn heap patterns into sorted vector
  std::vector<Pattern> result;
  result.reserve(patterns.size());

  while (!patterns.empty()) {
    result.push_back(patterns.top());
    patterns.pop();
  }
  if (sort_alpha) {
    std::sort(result.begin(), result.end(),
              [](const Pattern &a, const Pattern &b) {
                return a.pattern < b.pattern;
              });
  }
  return result;
}

Projection PatternMiner::project(DiscreteDB &ts, const Pattern &pattern) {
  Projection projection;
  int pattern_size = pattern.pattern.size();
  int current_max_size = static_cast<int>(pattern_size * duration);
  int current_max_gaps = current_max_size - static_cast<int>(pattern_size);
  for (int i = 0; i < ts.size(); ++i) {
    int gaps = 0;
    int symbol = 0;
    int start = 0;

    for (int j = 0; j < ts[i].size(); ++j) {
      // Current window item matches wanted symbol
      if (ts[i][j] == pattern.pattern[symbol]) {
        // First window item that matches symbol marks start of pattern
        if (symbol == 0) {
          start = j;
        }
        ++symbol;

        // Last window item that matches symbol marks end of pattern
        if (symbol == pattern_size) {
          projection[i] = {start, j + 1};
          break;
        }

      } else if (start < j) { // Gaps cannot occur before start of pattern
        ++gaps;
        if (gaps > current_max_gaps) {
          break;
        }
      }
    }
  }

  return projection;
}

Projection PatternMiner::project_soft(DiscreteDB &ts, const Pattern &pattern,
                                      double tau) {
  Projection projection;
  int pattern_size = pattern.pattern.size();
  double max_dist = pow(tau * pattern_size, 2);

  // Loop over windows
  for (int ts_idx = 0; ts_idx < ts.size(); ++ts_idx) {
    // Loop over potential window starting positions
    for (int start = 0; start < ts[ts_idx].size() - pattern_size; ++start) {
      double dist = 0;
      // Try to match pattern to this starting position
      for (int symbol = 0; symbol < pattern_size; ++symbol) {
        dist += pow(ts[ts_idx][start + symbol] - pattern.pattern[symbol], 2);

        // First symbols already exceed max_dist
        if (dist > max_dist) {
          break;
        }
      }

      // Match found; add to projection and stop looking in this window
      if (dist < max_dist) {
        projection[ts_idx] = {start, start + pattern_size};
        break;
      }
    }
  }

  return projection;
}

// --- Private Methods ---
void PatternMiner::mine_singletons(DiscreteDB &ts) {
  // Assume that every sax symbol occurs at least once
  for (int item = 0; item < alpha; ++item) {
    Word pattern = {item};
    auto projection = project_item(ts, item);
    auto candidates = get_candidates(ts, projection, pattern);
    queue.emplace(pattern, projection, candidates, projection.size());
  }
}

Projection PatternMiner::project_item(DiscreteDB &ts, int item) {
  Projection projection;
  for (size_t ts_idx = 0; ts_idx < ts.size(); ++ts_idx) {
    for (size_t j = 0; j + min_size <= ts[ts_idx].size(); ++j) {
      if (ts[ts_idx][j] == item) {
        projection[ts_idx] = {static_cast<int>(j), static_cast<int>(j + 1)};
        break;
      }
    }
  }
  return projection;
}

std::pair<Projection, int>
PatternMiner::project_incremental(DiscreteDB &ts, const Pattern &pattern,
                                  int item) {
  Projection projection;
  int support;
  for (const auto &[ts_idx, range] : pattern.projection) {
    int start = range.first, end = range.second;
    int current_max_size = start + max_duration;
    int current_max_gaps =
        static_cast<int>((pattern.pattern.size()) * (duration - 1));
    int current_gaps = end - start - pattern.pattern.size();
    int remaining_gaps = max_gaps - current_gaps;
    int stop = std::min({static_cast<int>(ts[ts_idx].size()), current_max_size,
                         end + remaining_gaps + 1});

    for (int j = end; j < stop; ++j) {
      if (ts[ts_idx][j] == item) {
        projection[ts_idx] = {start, j + 1};
        if ((j - start - pattern.pattern.size()) < current_max_gaps) {
          ++support;
        }
        break;
      }
    }
  }
  std::pair<Projection, int> result{projection, support};
  return result;
}

Candidates PatternMiner::get_candidates(DiscreteDB &ts,
                                        const Projection &projection,
                                        const Word &pattern) {
  Candidates candidates;
  for (const auto &[ts_idx, range] : projection) {
    if (candidates.size() == static_cast<size_t>(alpha))
      break;

    int start = range.first, end = range.second;
    int current_max_size = start + max_duration;
    int current_gaps = (end - start) - pattern.size();
    int remaining_gaps = max_gaps - current_gaps;
    int stop = std::min({static_cast<int>(ts[ts_idx].size()), current_max_size,
                         end + remaining_gaps + 1});

    for (int j = end; j < stop; ++j) {
      candidates.insert(ts[ts_idx][j]);
    }
  }
  return candidates;
}
