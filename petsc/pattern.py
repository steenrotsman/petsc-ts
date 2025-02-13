class Pattern:
    def __init__(self, pattern, projection, candidates):
        self.pattern = pattern
        self.projection = projection
        self.candidates = candidates

        # When instantiated, Pattern is added to queue, which is a max-heap
        self.support = -len(projection)

        # Ability to reproduce this pattern from original data
        self.signal = None
        self.window = None

    def __repr__(self):
        return f"Pattern({self.pattern}, {self.support})"

    def __len__(self):
        return len(self.pattern)

    def __eq__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.support == other.support

    def __lt__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.support < other.support
