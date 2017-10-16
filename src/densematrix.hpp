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

        std::vector<double> Mult(const std::vector<double>& input) const;
        std::vector<double> MultAT(const std::vector<double>& input) const;

        void Mult(const std::vector<double>& input, std::vector<double>& output) const;
        void MultAT(const std::vector<double>& input, std::vector<double>& output) const;

        DenseMatrix Mult(const DenseMatrix& input) const;
        DenseMatrix MultAT(const DenseMatrix& input) const;
        DenseMatrix MultBT(const DenseMatrix& input) const;
        DenseMatrix MultABT(const DenseMatrix& input) const;

        void Mult(const DenseMatrix& input, DenseMatrix& output) const;
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;
        void MultBT(const DenseMatrix& input, DenseMatrix& output) const;
        void MultABT(const DenseMatrix& input, DenseMatrix& output) const;

        DenseMatrix& operator-=(const DenseMatrix& other);
        DenseMatrix& operator+=(const DenseMatrix& other);

        friend DenseMatrix operator+(DenseMatrix lhs, const DenseMatrix& rhs);
        friend DenseMatrix operator-(DenseMatrix lhs, const DenseMatrix& rhs);

        DenseMatrix& operator*=(double val);
        friend DenseMatrix operator*(DenseMatrix lhs, double val);
        friend DenseMatrix operator*(double val, DenseMatrix rhs);

        DenseMatrix& operator/=(double val);
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

} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
