#include "pattern.h"

Pattern::Pattern(Word pattern, Projection projection,
                 Candidates candidates, int support)
    : pattern(std::move(pattern)), projection(std::move(projection)),
      candidates(std::move(candidates)), support(support) {
  coef = 0.0;
}

bool Pattern::operator<(const Pattern &other) const {
  return support < other.support;
}

bool Pattern::operator>(const Pattern &other) const {
  return support > other.support;
}
