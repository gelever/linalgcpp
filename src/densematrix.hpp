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
        explicit DenseMatrix(size_t size);

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
        DenseMatrix(const DenseMatrix& other) noexcept;

        /*! @brief Move constructor */
        DenseMatrix(DenseMatrix&& other) noexcept;

        /*! @brief Set this matrix equal to other */
        DenseMatrix& operator=(DenseMatrix other) noexcept;

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
            @param width total width of each entry including negative
            @param precision precision to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout, int width = 8, int precision = 4) const;

        /*! @brief Get the transpose of this matrix
            @retval transpose the tranpose
        */
        DenseMatrix Transpose() const;

        /*! @brief Store the transpose of this matrix
                   into the user provided matrix
            @param transpose transpose the tranpose

            @note Size must be set beforehand!
        */
        void Transpose(DenseMatrix& transpose) const;

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
        Vector<double> Mult(const VectorView<T>& input) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T>
        Vector<double> MultAT(const VectorView<T>& input) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T, typename T2>
        void Mult(const VectorView<T>& input, VectorView<T2>& output) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T, typename T2>
        void MultAT(const VectorView<T>& input, VectorView<T2>& output) const;

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

        /*! @brief Check if the dense matrices are equal
            @param other the other DenseMatrix
            @retval true if the dense matrices are close enough to equal
        */
        bool operator==(const DenseMatrix& other) const;

        /*! @brief Get a single column from the matrix
            @param col the column to get
            @param vect set this vect to the column values
        */
        template <typename T = double>
        void GetCol(size_t col, VectorView<T>& vect) const;

        /*! @brief Get a single row from the matrix
            @param row the row to get
            @param vect set this vect to the row values
        */
        template <typename T = double>
        void GetRow(size_t row, VectorView<T>& vect) const;

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
        void SetCol(size_t col, const VectorView<T>& vect);

        /*! @brief Set a single row vector's values
            @param row the row to set
            @param vect the values to set
        */
        template <typename T = double>
        void SetRow(size_t row, const VectorView<T>& vect);

        /*! @brief Get a range of rows from the matrix
            @param start start of range, inclusive
            @param end end of range, inclusive
            @retval DenseMatrix the range of rows
        */
        DenseMatrix GetRow(size_t start, size_t end) const;

        /*! @brief Get a range of rows from the matrix
            @param start start of range, inclusive
            @param end end of range, inclusive
            @param dense dense matrix that will hold the range
        */
        void GetRow(size_t start, size_t end, DenseMatrix& dense) const;

        /*! @brief Get a selection of rows from the matrix
            @param rows rows to select
            @retval DenseMatrix the selection of rows
        */
        DenseMatrix GetRow(const std::vector<int>& rows) const;

        /*! @brief Get a selection of rows from the matrix
            @param rows rows to select
            @param dense dense matrix that will hold the selection
        */
        void GetRow(const std::vector<int>& rows, DenseMatrix& dense) const;

        /*! @brief Set a range of rows from the matrix
            @param start start of range, inclusive
            @param dense dense matrix that holds the range
        */
        void SetRow(size_t start, const DenseMatrix& dense);

        /*! @brief Get a range of columns from the matrix
            @param start start of range, inclusive
            @param end end of range, exclusive
            @retval DenseMatrix the range of columns
        */
        DenseMatrix GetCol(size_t start, size_t end) const;

        /*! @brief Get a range of columns from the matrix
            @param start start of range, inclusive
            @param end end of range, exclusive
            @param dense dense matrix that will hold the range
        */
        void GetCol(size_t start, size_t end, DenseMatrix& dense) const;

        /*! @brief Set a range of columns from the matrix
            @param start start of range, inclusive
            @param dense dense matrix that holds the range
        */
        void SetCol(size_t start, const DenseMatrix& dense);

        /*! @brief Get a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @retval dense dense matrix that will hold the range
        */
        DenseMatrix GetSubMatrix(size_t start_i, size_t start_j, size_t end_i, size_t end_j) const;

        /*! @brief Get a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @param dense dense matrix that will hold the range
        */
        void GetSubMatrix(size_t start_i, size_t start_j, size_t end_i, size_t end_j, DenseMatrix& dense) const;

        /*! @brief Set a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @param dense dense matrix that holds the range
        */
        void SetSubMatrix(size_t start_i, size_t start_j, size_t end_i, size_t end_j, const DenseMatrix& dense);

        /*! @brief Solve eigenvalue problem AQ = LQ,
                   where Q are eigenvectors and L is eigenvalues
                   A is replaced with the computed eigenvectors
            @warning this replaces this matrix with the eigenvectors!
            @returns eigenvalues the computed eigenvalues
        */
        std::vector<double> EigenSolve();

        /*! @brief Solve eigenvalue problem AQ = LQ,
                   where Q are eigenvectors and L is eigenvalues
            @param[out] eigenvectors DenseMatrix to hold the computed eigenvectors
            @returns eigenvalues the computed eigenvalues
        */
        std::vector<double> EigenSolve(DenseMatrix& eigenvectors) const;

        /*! @brief Compute singular values and vectors A = U * S * VT
                   Where S is returned and A is replaced with VT
            @warning this replaces this matrix with U!
            @returns singular_values the computed singular_values
        */
        std::vector<double> SVD();

        /*! @brief Compute singular values and vectors A = U * S * VT
                   Where S is returned and A is replaced with U
            @param[out] VT DenseMatrix to hold the computed U
            @returns singular_values the computed singular_values
        */
        std::vector<double> SVD(DenseMatrix& U) const;

        /*! @brief Compute QR decomposition
            @warning this replaces this matrix with Q!
        */
        void QR();

        /*! @brief Compute QR decomposition
            @param Q Stores Q instead of overwriting
        */
        void QR(DenseMatrix& Q) const;

        /*! @brief Scale rows by given values
            @param values scale per row
        */
        void ScaleRows(const std::vector<double>& values);

        /*! @brief Scale cols by given values
            @param values scale per cols
        */
        void ScaleCols(const std::vector<double>& values);

        /*! @brief Get Diagonal entries
            @returns diag diagonal entries
        */
        std::vector<double> GetDiag() const;

        /*! @brief Get Diagonal entries
            @param array to hold diag diagonal entries
        */
        void GetDiag(std::vector<double>& diag) const;

        /// Operator Requirement, calls the templated Mult
        void Mult(const VectorView<double>& input, VectorView<double>& output) const override;
        /// Operator Requirement, calls the templated MultAT
        void MultAT(const VectorView<double>& input, VectorView<double>& output) const override;

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

    double sum = std::accumulate(begin(data_), end(data_), 0.0);
    return sum;
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
Vector<double> DenseMatrix::Mult(const VectorView<T>& input) const
{
    Vector<double> output(rows_);
    Mult(input, output);

    return output;
}

template <typename T, typename T2>
void DenseMatrix::Mult(const VectorView<T>& input, VectorView<T2>& output) const
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
Vector<double> DenseMatrix::MultAT(const VectorView<T>& input) const
{
    Vector<double> output(cols_);
    MultAT(input, output);

    return output;
}

template <typename T, typename T2>
void DenseMatrix::MultAT(const VectorView<T>& input, VectorView<T2>& output) const
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
void DenseMatrix::GetCol(size_t col, VectorView<T>& vect) const
{
    assert(col >= 0 && col < cols_);
    assert(vect.size() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        vect[i] = (*this)(i, col);
    }
}

template <typename T>
void DenseMatrix::GetRow(size_t row, VectorView<T>& vect) const
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
    GetRow(row, vect);

    return vect;
}

template <typename T>
void DenseMatrix::SetCol(size_t col, const VectorView<T>& vect)
{
    assert(col >= 0 && col < cols_);
    assert(vect.size() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        (*this)(i, col) = vect[i];
    }
}

template <typename T>
void DenseMatrix::SetRow(size_t row, const VectorView<T>& vect)
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
