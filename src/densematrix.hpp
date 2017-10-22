/*! @file */

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

/*! @brief Dense matrix in column major format.
    This means data is arranged contiguously by column vectors.
*/
class DenseMatrix : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        DenseMatrix();

        /*! @brief Square Constructor of setting the number of rows
            and columns
            @param size the number of rows and columns
        */
        DenseMatrix(size_t size);

        /*! @brief Rectangle Constructor of setting the number of rows
            and columns
            @param rows the number of rows
            @param cols the number of columns

            @note values intitialed to zero
        */
        DenseMatrix(size_t rows, size_t cols);

        /*! @brief Rectangle Constructor of setting the number of rows,
            columns, and intital values
            @param rows the number of rows
            @param cols the number of columns
            @param data the initial data
        */
        DenseMatrix(size_t rows, size_t cols, const std::vector<double>& data);

        /*! @brief Copy Constructor */
        DenseMatrix(const DenseMatrix&) = default;

        /*! @brief Move constructor */
        DenseMatrix(DenseMatrix&&);

        /*! @brief Destructor */
        ~DenseMatrix() noexcept = default;

        /*! @brief Swap two matrices
            @param lhs left hand side matrix
            @param rhs right hand side matrix
        */
        friend void Swap(DenseMatrix& lhs, DenseMatrix& rhs);

        /*! @brief The number of rows in this matrix
            @retval the number of rows
        */
        size_t Rows() const;

        /*! @brief The number of columns in this matrix
            @retval the number of columns
        */
        size_t Cols() const;

        /*! @brief Computes the sum of all entries
            @retval the sum of all entries
        */
        double Sum() const;

        /*! @brief Computes the maximum of all entries
            @retval the maximum of all entries
        */
        double Max() const;

        /*! @brief Computes the minimum of all entries
            @retval the minimum of all entries
        */
        double Min() const;

        /*! @brief Print the entries of this matrix in dense format
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Index operator
            @param row row index
            @param col column index
            @retval a reference to the value at (i, j)
        */
        double& operator()(size_t row, size_t col);

        /*! @brief Const index operator
            @param row row index
            @param col column index
            @retval a const reference to the value at (i, j)
        */
        const double& operator()(size_t row, size_t col) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T>
        Vector<double> Mult(const Vector<T>& input) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T>
        Vector<double> MultAT(const Vector<T>& input) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T, typename T2>
        void Mult(const Vector<T>& input, Vector<T2>& output) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T, typename T2>
        void MultAT(const Vector<T>& input, Vector<T2>& output) const;

        /*! @brief Multiplies a dense matrix: AB = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix Mult(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: A^T B = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultAT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: AB^T = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultBT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: A^T B^T = Y
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultABT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: AB = C
            @param input the input dense matrix B
            @param output the output dense matrix C
        */
        void Mult(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix: A^T B = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix: AB^T = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        void MultBT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix: A^T B^T = C
            @param input the input dense matrix B
            @param output the output dense matrix C
        */
        void MultABT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Adds the entries of another matrix to this one
            @param other the other dense matrix
        */
        DenseMatrix& operator+=(const DenseMatrix& other);

        /*! @brief Subtracts the entries of another matrix to this one
            @param other the other dense matrix
        */
        DenseMatrix& operator-=(const DenseMatrix& other);

        /*! @brief Multiply by a scalar */
        DenseMatrix& operator*=(double val);

        /*! @brief Divide by a scalar */
        DenseMatrix& operator/=(double val);

        /*! @brief Adds two matrices together A + B = C
            @param lhs the left hand side matrix A
            @param rhs the right hand side matrix B
            @retval The sum of the matrices C
        */
        friend DenseMatrix operator+(DenseMatrix lhs, const DenseMatrix& rhs);

        /*! @brief Subtract two matrices A - B = C
            @param lhs the left hand side matrix A
            @param rhs the right hand side matrix B
            @retval The difference of the matrices C
        */
        friend DenseMatrix operator-(DenseMatrix lhs, const DenseMatrix& rhs);

        /*! @brief Multiply a matrix by a scalar Aa = C
            @param lhs the left hand side matrix A
            @param val the scalar a
            @retval The multiplied matrix C
        */
        friend DenseMatrix operator*(DenseMatrix lhs, double val);

        /*! @brief Multiply a matrix by a scalar aA = C
            @param val the scalar a
            @param rhs the right hand side matrix A
            @retval The multiplied matrix C
        */
        friend DenseMatrix operator*(double val, DenseMatrix rhs);

        /*! @brief Divide all entries a matrix by a scalar A/a = C
            @param lhs the left hand side matrix A
            @param val the scalar a
            @retval The divided matrix C
        */
        friend DenseMatrix operator/(DenseMatrix lhs, double val);

        /*! @brief Divide a scalar by all entries of a matrix a/A = C
            @param val the scalar a
            @param rhs the right hand side matrix A
            @retval The divided matrix C
        */
        friend DenseMatrix operator/(double val, DenseMatrix rhs);

        /*! @brief Set all entries to a scalar value
            @param val the scalar a
        */
        DenseMatrix& operator=(double val);

        /// Operator Requirement, calls the templated Mult
        void Mult(const Vector<double>& input, Vector<double>& output) const override;
        /// Operator Requirement, calls the templated MultAT
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


} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
