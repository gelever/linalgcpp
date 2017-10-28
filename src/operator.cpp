/* @file */

#include "operator.hpp"

namespace linalgcpp
{

Vector<double> Operator::Mult(const Vector<double>& input) const
{
    Vector<double> output(Rows());
    Mult(input, output);
    
    return output;
}

Vector<double> Operator::MultAT(const Vector<double>& input) const
{
    Vector<double> output(Rows());
    MultAT(input, output);
    
    return output;
}

double Operator::InnerProduct(const Vector<double>& x, const Vector<double>& y) const
{
    return y * Mult(x);
}

} // namespace linalgcpp
