import heapq

from .pattern import Pattern


class PatternMiner:
    def __init__(self, alpha, min_size, max_size, duration, k, sort_alpha):
        self.alpha = alpha
        self.min_size = min_size
        self.max_size = max_size
        self.duration = duration
        self.max_duration = int(self.max_size * duration)
        self.max_gaps = self.max_duration - self.max_size
        self.k = k
        self.sort_alpha = sort_alpha

        self.patterns_ = []
        self.n_ = 0

    def mine(self, ts):
        queue = []
        # Level 1: singletons; assume that every sax symbol occurs at least once
        for item in range(self.alpha):
            pattern = [item]
            projection = self.compute_projection_singleton(ts, item)
            candidates = self.get_candidates(ts, projection, pattern)
            heapq.heappush(queue, Pattern(pattern, projection, candidates))

        # Levels n+1
        while queue:
            pattern = heapq.heappop(queue)

            # All other candidates have support <= pattern.support
            if self.n_ == self.k and -pattern.support < self.patterns_[0].support:
                break

            # Only generate candidates if they are short enough
            if len(pattern) < self.max_size:
                for item in pattern.candidates:
                    candidate = pattern.pattern + [int(item)]
                    projection = self.compute_projection_incremental(ts, pattern, item)

                    # Don't add candidate if its support is lower than kth patttern
                    if self.n_ < self.k or len(projection) > self.patterns_[0].support:
                        candidates = self.get_candidates(ts, projection, candidate)
                        p = Pattern(candidate, projection, candidates)
                        heapq.heappush(queue, p)

            # Reverse ordering because queue is max-heap and self.patterns is min-heap
            pattern.support *= -1

            # Add pattern to self.patterns if length and support are high enough
            if len(pattern) < self.min_size:
                continue
            elif self.n_ < self.k:
                heapq.heappush(self.patterns_, pattern)
                self.n_ += 1
            else:
                heapq.heappushpop(self.patterns_, pattern)
                self.n_ += 1

        if self.sort_alpha:
            return sorted(self.patterns_, key=lambda p: p.pattern)
        else:
            return sorted(self.patterns_, reverse=True)

    def compute_projection_singleton(self, ts, item):
        projection = {}
        for ts_idx, current_ts in enumerate(ts):
            for j in range(len(current_ts) - self.min_size + 1):
                if current_ts[j] == item:
                    projection[ts_idx] = (j, j + 1)
                    break  # Only keep the first occurrence
        return projection

    def compute_projection_incremental(self, ts, pattern, item):
        """Note: this method is 80-90% of run time."""
        p = {}
        for ts_idx, (start, end) in pattern.projection.items():
            max_size = start + self.max_duration
            current_gaps = (end - start) - len(pattern.pattern)
            remaining_gaps = self.max_gaps - current_gaps
            stop = min(len(ts[ts_idx]), max_size, end + remaining_gaps + 1)

            for j in range(end, stop):
                if ts[ts_idx][j] == item:
                    p[ts_idx] = (start, j + 1)
                    break  # Only project on first occurrence

        return p

    def get_candidates(self, ts, projection, pattern):
        candidates = set()

        for ts_idx, (start, end) in projection.items():
            # If entire alphabet is already candidate, no more can be added
            if len(candidates) == self.alpha:
                break

            max_size = start + self.max_duration
            current_gaps = (end - start) - len(pattern)
            remaining_gaps = self.max_gaps - current_gaps
            stop = min(len(ts[ts_idx]), max_size, end + remaining_gaps + 1)

            for j in range(end, stop):
                candidates.add(ts[ts_idx][j])

        return candidates
