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
        DenseMatrix(DenseMatrix&&);

        ~DenseMatrix() noexcept = default;

        friend void Swap(DenseMatrix& lhs, DenseMatrix& rhs);

        size_t Rows() const;
        size_t Cols() const;

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

        /*! @brief Get a single column from the matrix
            @param col the column to get
            @param vect set this vect to the column values
        */
        template <typename T = double>
        void GetCol(size_t col, Vector<T>& vect) const;

        /*! @brief Get a single row from the matrix
            @param row the row to get
            @param vect set this vect to the row values
        */
        template <typename T = double>
        void GetRow(size_t row, Vector<T>& vect) const;

        /*! @brief Get a single column from the matrix
            @param col the column to get
            @retval vect the vect of column values
        */
        template <typename T = double>
        Vector<T> GetCol(size_t col) const;

        /*! @brief Get a single row from the matrix
            @param row the row to get
            @retval vect the vect of row values
        */
        template <typename T = double>
        Vector<T> GetRow(size_t row) const;

        /*! @brief Set a single column vector's values
            @param col the column to set
            @param vect the values to set
        */
        template <typename T = double>
        void SetCol(size_t col, const Vector<T>& vect);

        /*! @brief Set a single row vector's values
            @param row the row to set
            @param vect the values to set
        */
        template <typename T = double>
        void SetRow(size_t row, const Vector<T>& vect);

        // Operator Requirement
        void Mult(const Vector<double>& input, Vector<double>& output) const override;
        void MultAT(const Vector<double>& input, Vector<double>& output) const override;

    private:
        size_t rows_;
        size_t cols_;
        std::vector<double> data_;

        void dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const;

};

inline
double& DenseMatrix::operator()(size_t row, size_t col)
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < rows_);
    assert(col < cols_);

    return data_[row + (col * rows_)];
}

inline
const double& DenseMatrix::operator()(size_t row, size_t col) const
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < rows_);
    assert(col < cols_);

    return data_[row + (col * rows_)];
}

inline
size_t DenseMatrix::Rows() const
{
    return rows_;
}

inline
size_t DenseMatrix::Cols() const
{
    return cols_;
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

    for (size_t j = 0; j < cols_; ++j)
    {
        for (size_t i = 0; i < rows_; ++i)
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

    for (size_t j = 0; j < cols_; ++j)
    {
        T2 val = 0;

        for (size_t i = 0; i < rows_; ++i)
        {
            val += (*this)(i, j) * input[i];
        }

        output[j] = val;
    }
}

template <typename T>
void DenseMatrix::GetCol(size_t col, Vector<T>& vect) const
{
    assert(col >= 0 && col < cols_);
    assert(vect.size() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        vect[i] = (*this)(i, col);
    }
}

template <typename T>
void DenseMatrix::GetRow(size_t row, Vector<T>& vect) const
{
    assert(row >= 0 && row < rows_);
    assert(vect.size() == cols_);

    for (size_t i = 0; i < cols_; ++i)
    {
        vect[i] = (*this)(row, i);
    }
}

template <typename T>
Vector<T> DenseMatrix::GetCol(size_t col) const
{
    Vector<T> vect(rows_);
    GetCol(col, vect);

    return vect;
}

template <typename T>
Vector<T> DenseMatrix::GetRow(size_t row) const
{
    Vector<T> vect(cols_);
    GetCol(row, vect);

    return vect;
}

template <typename T>
void DenseMatrix::SetCol(size_t col, const Vector<T>& vect)
{
    assert(col >= 0 && col < cols_);
    assert(vect.size() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        (*this)(i, col) = vect[i];
    }
}

template <typename T>
void DenseMatrix::SetRow(size_t row, const Vector<T>& vect)
{
    assert(row >= 0 && row < rows_);
    assert(vect.size() == cols_);

    for (size_t i = 0; i < cols_; ++i)
    {
        (*this)(row, i) = vect[i];
    }
}

} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
