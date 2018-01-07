/* @file */

#include "operator.hpp"

namespace linalgcpp
{

Vector<double> Operator::Mult(const VectorView<double>& input) const
{
    Vector<double> output(Rows());
    Mult(input, output);

    return output;
}

Vector<double> Operator::MultAT(const VectorView<double>& input) const
{
    Vector<double> output(Rows());
    MultAT(input, output);

    return output;
}

double Operator::InnerProduct(const VectorView<double>& x, const VectorView<double>& y) const
{
    return y.Mult(Mult(x));
}

} // namespace linalgcpp
