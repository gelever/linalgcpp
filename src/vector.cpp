#include "vector.hpp"


void Normalize(std::vector<double>& vect)
{
	vect /= L2Norm(vect);
}

double L2Norm(const std::vector<double>& vect)
{
	return std::sqrt(vect * vect);
}

void SubAvg(std::vector<double>& vect)
{
	vect -= Mean(vect);
}

