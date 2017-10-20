#ifndef DENSEMATRIX_HPP__
#define DENSEMATRIX_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "operator.hpp"
#include "vector.hpp"

namespace linalgcpp
{

class DenseMatrix : public Operator
{
    public:
        DenseMatrix();
        DenseMatrix(size_t size);
        DenseMatrix(size_t rows, size_t cols);
        DenseMatrix(size_t rows, size_t cols, const std::vector<double>& data);

        DenseMatrix(const DenseMatrix&) = default;

        friend void Swap(DenseMatrix& lhs, DenseMatrix& rhs);
        DenseMatrix(DenseMatrix&&);
        ~DenseMatrix() noexcept = default;

        double Sum() const;
        double Max() const;
        double Min() const;

        void Print(const std::string& label = "") const;

        double& operator()(size_t row, size_t col);
        const double& operator()(size_t row, size_t col) const;

        template <typename T>
        Vector<double> Mult(const Vector<T>& input) const;

        template <typename T>
        Vector<double> MultAT(const Vector<T>& input) const;

        template <typename T, typename T2>
        void Mult(const Vector<T>& input, Vector<T2>& output) const;

        template <typename T, typename T2>
        void MultAT(const Vector<T>& input, Vector<T2>& output) const;

        DenseMatrix Mult(const DenseMatrix& input) const;
        DenseMatrix MultAT(const DenseMatrix& input) const;
        DenseMatrix MultBT(const DenseMatrix& input) const;
        DenseMatrix MultABT(const DenseMatrix& input) const;

        void Mult(const DenseMatrix& input, DenseMatrix& output) const;
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;
        void MultBT(const DenseMatrix& input, DenseMatrix& output) const;
        void MultABT(const DenseMatrix& input, DenseMatrix& output) const;

        DenseMatrix& operator+=(const DenseMatrix& other);
        DenseMatrix& operator-=(const DenseMatrix& other);
        DenseMatrix& operator*=(double val);
        DenseMatrix& operator/=(double val);

        friend DenseMatrix operator+(DenseMatrix lhs, const DenseMatrix& rhs);
        friend DenseMatrix operator-(DenseMatrix lhs, const DenseMatrix& rhs);
        friend DenseMatrix operator*(DenseMatrix lhs, double val);
        friend DenseMatrix operator*(double val, DenseMatrix rhs);
        friend DenseMatrix operator/(DenseMatrix lhs, double val);

        DenseMatrix& operator=(double val);

        // Operator Requirement
        void Mult(const Vector<double>& input, Vector<double>& output) const
        {
            Mult<double, double>(input, output);
        }

    private:
        std::vector<double> data_;

        void dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const;

};

inline
double& DenseMatrix::operator()(size_t row, size_t col)
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < Rows());
    assert(col < Cols());

    return data_[row + (col * Rows())];
}

inline
const double& DenseMatrix::operator()(size_t row, size_t col) const
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < Rows());
    assert(col < Cols());

    return data_[row + (col * Rows())];
}

inline
double DenseMatrix::Sum() const
{
    assert(data_.size());

    double total = 0.0;
    std::accumulate(begin(data_), end(data_), total);

    return total;
}

inline
double DenseMatrix::Max() const
{
    assert(data_.size());

    return *std::max_element(begin(data_), end(data_));
}

inline
double DenseMatrix::Min() const
{
    assert(data_.size());

    return *std::min_element(begin(data_), end(data_));
}

template <typename T>
Vector<double> DenseMatrix::Mult(const Vector<T>& input) const
{
    Vector<double> output(Rows());
    Mult(input, output);

    return output;
}

template <typename T, typename T2>
void DenseMatrix::Mult(const Vector<T>& input, Vector<T2>& output) const
{
    assert(input.size() == Cols());
    assert(output.size() == Rows());

    output = 0;

    for (size_t j = 0; j < Cols(); ++j)
    {
        for (size_t i = 0; i < Rows(); ++i)
        {
            output[i] += (*this)(i, j) * input[j];
        }
    }
}

template <typename T>
Vector<double> DenseMatrix::MultAT(const Vector<T>& input) const
{
    Vector<double> output(Cols());
    MultAT(input, output);

    return output;
}

template <typename T, typename T2>
void DenseMatrix::MultAT(const Vector<T>& input, Vector<T2>& output) const
{
    assert(input.size() == Rows());
    assert(output.size() == Cols());

    for (size_t j = 0; j < Cols(); ++j)
    {
        T2 val = 0;

        for (size_t i = 0; i < Rows(); ++i)
        {
            val += (*this)(i, j) * input[i];
        }

        output[j] = val;
    }
}


} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
