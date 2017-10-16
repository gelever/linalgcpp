#include "vector.hpp"

namespace linalgcpp
{

void Normalize(Vector<double>& vect)
{
    vect /= L2Norm(vect);
}

void SubAvg(Vector<double>& vect)
{
    vect -= Mean(vect);
}

template class Vector<int>;
template class Vector<double>;

} // namespace linalgcpp
