#ifndef OPERATOR_HPP__
#define OPERATOR_HPP__

#include <memory>
#include <vector>
#include <numeric>
#include <assert.h>

#include "vector.hpp"

namespace linalgcpp
{

class Operator
{
    public:
        Operator();
        Operator(size_t size);
        Operator(size_t rows, size_t cols);
        Operator(const Operator& other) = default;
        Operator(Operator&& other) = default;

        virtual size_t Rows() const;
        virtual size_t Cols() const;

        friend void Swap(Operator& lhs, Operator& rhs);

        virtual ~Operator() noexcept = default;

        virtual void Mult(const Vector<double>& input, Vector<double>& output) const = 0;

    private:
        size_t rows_;
        size_t cols_;
};

}
#endif // OPERATOR_HPP__
