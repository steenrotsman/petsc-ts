#ifndef PATTERN_H
#define PATTERN_H

#include <vector>

#include "typing.h"

class Pattern {
public:
  Word pattern;
  Projection projection;
  Candidates candidates;
  int support;
  double coef;

  Pattern(Word pattern, Projection projection,
          Candidates candidates, int support);
  bool operator<(const Pattern &other) const;
  bool operator>(const Pattern &other) const;
};

#endif
