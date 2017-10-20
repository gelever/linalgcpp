#include "operator.hpp"

namespace linalgcpp
{

Operator::Operator() : Operator(0)
{

}

Operator::Operator(size_t size) : Operator(size, size)
{

}

Operator::Operator(size_t rows, size_t cols)
    : rows_(rows), cols_(cols)
{

}

size_t Operator::Rows() const
{
    return rows_;
}

size_t Operator::Cols() const
{
    return cols_;
}

void Swap(Operator& lhs, Operator& rhs)
{
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.cols_, rhs.cols_);
}



}
