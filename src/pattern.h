#ifndef PATTERN_H
#define PATTERN_H

#include <vector>

#include "typing.h"

class Pattern
{
    public:
        std::vector<int> pattern;
        Projection projection;
        Candidates candidates;
        int support;
        double coef;

        Pattern(std::vector<int> pattern, Projection projection, Candidates candidates);
        bool operator<(const Pattern &other) const;
        bool operator>(const Pattern &other) const;
};

#endif