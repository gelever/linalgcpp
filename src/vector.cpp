#include "vector.hpp"

namespace linalgcpp
{
void Randomize(Vector<double>& vect, double lo, double hi)
{
    std::random_device r;
    std::default_random_engine dev(r());
    std::uniform_real_distribution<double> gen(lo, hi);

    for (double& val : vect)
    {
        val = gen(dev);
    }
}

void Randomize(Vector<int>& vect, int lo, int hi)
{
    std::random_device r;
    std::default_random_engine dev(r());
    std::uniform_int_distribution<int> gen(lo, hi);

    for (int& val : vect)
    {
        val = gen(dev);
    }
}

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
