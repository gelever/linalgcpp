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

#include "vector.hpp"

namespace linalgcpp
{

class DenseMatrix
{
    public:
        DenseMatrix();
        DenseMatrix(int size);
        DenseMatrix(int rows, int cols);
        DenseMatrix(int rows, int cols, const std::vector<double>& data);

        DenseMatrix(const DenseMatrix&) = default;

        friend void Swap(DenseMatrix& lhs, DenseMatrix& rhs);
        DenseMatrix(DenseMatrix&&);
        ~DenseMatrix() noexcept = default;

        int Rows() const;
        int Cols() const;

        double Sum() const;
        double Max() const;
        double Min() const;

        void Print(const std::string& label = "") const;

        double& operator()(int row, int col);
        const double& operator()(int row, int col) const;

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

    private:
        int rows_;
        int cols_;
        std::vector<double> data_;

        void dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const;

};

inline
int DenseMatrix::Rows() const
{
    return rows_;
}

inline
int DenseMatrix::Cols() const
{
    return cols_;
}

inline
double& DenseMatrix::operator()(int row, int col)
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < rows_);
    assert(col < cols_);

    return data_[row + (col * rows_)];
}

inline
const double& DenseMatrix::operator()(int row, int col) const
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < rows_);
    assert(col < cols_);

    return data_[row + (col * rows_)];
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
    Vector<double> output(rows_);
    Mult(input, output);

    return output;
}

template <typename T, typename T2>
void DenseMatrix::Mult(const Vector<T>& input, Vector<T2>& output) const
{
    assert(input.size() == cols_);
    assert(output.size() == rows_);

    output = 0;

    for (int j = 0; j < cols_; ++j)
    {
        for (int i = 0; i < rows_; ++i)
        {
            output[i] += (*this)(i, j) * input[j];
        }
    }
}

template <typename T>
Vector<double> DenseMatrix::MultAT(const Vector<T>& input) const
{
    Vector<double> output(cols_);
    MultAT(input, output);

    return output;
}

template <typename T, typename T2>
void DenseMatrix::MultAT(const Vector<T>& input, Vector<T2>& output) const
{
    assert(input.size() == rows_);
    assert(output.size() == cols_);

    for (int j = 0; j < cols_; ++j)
    {
        T2 val = 0;

        for (int i = 0; i < rows_; ++i)
        {
            val += (*this)(i, j) * input[i];
        }

        output[j] = val;
    }
}


} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
