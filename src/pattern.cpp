#include "pattern.h"

Pattern::Pattern(std::vector<int> pattern, Projection projection, Candidates candidates) : pattern(std::move(pattern)), projection(std::move(projection)), candidates(std::move(candidates))
{
    support = static_cast<int>(projection.size());
    coef = 0.0;
}

bool Pattern::operator<(const Pattern &other) const
{
    return support < other.support;
}

bool Pattern::operator>(const Pattern &other) const
{
    return support > other.support;
}
