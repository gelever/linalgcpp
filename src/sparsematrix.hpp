/*! @file */

#ifndef SPARSEMATRIX_HPP__
#define SPARSEMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

#include "operator.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

/*! @brief Sparse matrix in CSR format

    3 arrays keep track of the nonzero entries:
        - indptr is the row pointer such that indptr[i] points
          to the start of row i in indices and data
        - indices is the column index of the entry
        - data is the value of the entry
*/
template <typename T = double>
class SparseMatrix : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        SparseMatrix();

        /*! @brief Empty Constructor with set size
            @param size rows and columns in matrix
        */
        explicit SparseMatrix(int size);

        /*! @brief Empty Constructor with set size
            @param rows the number of rows
            @param cols the number of columns
        */
        SparseMatrix(int rows, int cols);

        /*! @brief Constructor setting the individual arrays and size
            @param indptr row pointer array
            @param indices column indices array
            @param data entry value array
            @param rows number of rows in the matrix
            @param cols number of cols in the matrix
        */
        SparseMatrix(const std::vector<int>& indptr,
                     const std::vector<int>& indices,
                     const std::vector<T>& data,
                     int rows, int cols);

        /*! @brief Diagonal Constructor
            @param diag values for the diagonal
        */
        explicit SparseMatrix(const std::vector<T>& diag);

        /*! @brief Copy Constructor */
        SparseMatrix(const SparseMatrix<T>& other) noexcept;

        /*! @brief Move constructor */
        SparseMatrix(SparseMatrix<T>&& other) noexcept;

        /*! @brief Destructor */
        ~SparseMatrix() noexcept = default;

        /*! @brief Sets this matrix equal to another
            @param other the matrix to copy
        */
        SparseMatrix<T>& operator=(SparseMatrix<T> other) noexcept;

        /*! @brief Swap two matrices
            @param lhs left hand side matrix
            @param rhs right hand side matrix
        */
        template <typename T2>
        friend void Swap(SparseMatrix<T2>& lhs, SparseMatrix<T2>& rhs);

        /*! @brief The number of rows in this matrix
            @retval the number of rows
        */
        size_t Rows() const override;

        /*! @brief The number of columns in this matrix
            @retval the number of columns
        */
        size_t Cols() const override;

        /*! @brief The number of nonzero entries in this matrix
            @retval the nonzero entries of columns

            @note this includes explicit zeros
        */
        size_t nnz() const;

        /*! @brief Get the const row pointer array
            @retval the row pointer array
        */
        const std::vector<int>& GetIndptr() const;

        /*! @brief Get the const column indices
            @retval the column indices array
        */
        const std::vector<int>& GetIndices() const;

        /*! @brief Get the const entry values
            @retval the data array
        */
        const std::vector<T>& GetData() const;

        /*! @brief Get the row pointer array
            @retval the row pointer array
        */
        std::vector<int>& GetIndptr();

        /*! @brief Get the column indices
            @retval the column indices array
        */
        std::vector<int>& GetIndices();

        /*! @brief Get the entry values
            @retval the data array
        */
        std::vector<T>& GetData();

        /*! @brief Get the indices from one row
            @param row the row to get
            @retval indices the indices from one row
        */
        std::vector<int> GetIndices(size_t row) const;

        /*! @brief Get the entries from one row
            @param row the row to get
            @retval the data from one row
        */
        std::vector<T> GetData(size_t row) const;

        /*! @brief Get the number of entries in a row
            @param row the row to get
            @retval size_t the number of entries in the row
        */
        size_t RowSize(size_t row) const;

        /*! @brief Print the nonzero entries as a list
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Print the entries of this matrix in dense format
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        void PrintDense(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Generate a dense version of this matrix
            @retval the dense version of this matrix
        */
        DenseMatrix ToDense() const;

        /*! @brief Sort the column indices in each row */
        void SortIndices();

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T2 = T>
        auto Mult(const VectorView<T2>& input) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T2 = T>
        auto MultAT(const VectorView<T2>& input) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T2 = T, typename T3 = T>
        void Mult(const VectorView<T2>& input, VectorView<T3>& output) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T2 = T, typename T3 = T>
        void MultAT(const VectorView<T2>& input, VectorView<T3>& output) const;

        /*! @brief Multiplies a dense matrix: AX = Y
            @param input the input dense matrix X
            @retval output the output dense matrix Y
        */
        DenseMatrix Mult(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix by the transpose
            of this matrix: A^T X = Y
            @param input the input dense matrix X
            @retval output the output dense matrix Y
        */
        DenseMatrix MultAT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: AX = Y
            @param input the input dense matrix X
            @retval output the output dense matrix Y
        */
        void Mult(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix by the transpose
            of this matrix: A^T X = Y
            @param input the input dense matrix X
            @param output the output dense matrix Y
        */
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a sparse matrix: AB = C
            @param rhs the input sparse matrix B
            @retval the output sparse matrix C
        */
        template <typename T2 = T, typename T3 = typename std::common_type<T, T2>::type>
        auto Mult(const SparseMatrix<T2>& rhs) const;

        /*! @brief Genereates the transpose of this matrix*/
        SparseMatrix<T> Transpose() const;

        /*! @brief Get the diagonal entries
            @retval the diagonal entries
        */
        std::vector<double> GetDiag() const;

        /*! @brief Add to the diagonal
            @param diag the diagonal entries
        */
        void AddDiag(const std::vector<T>& diag);

        /*! @brief Add to the diagonal
            @param diag the diagonal entries
        */
        void AddDiag(T val);

        /*! @brief Extract a submatrix out of this matrix
            @param rows the rows to extract
            @param cols the columns to extract
            @retval the submatrix

            @note a workspace array the size of the number of columns
                 in this matrix is used for this routine. If multiple
                 extractions are need from large matrices, it is
                 recommended to reuse a single marker array to avoid
                 large memory allocations.
        */
        SparseMatrix<T> GetSubMatrix(const std::vector<int>& rows,
                                     const std::vector<int>& cols) const;

        /*! @brief Extract a submatrix out of this matrix
            @param rows the rows to extract
            @param cols the columns to extract
            @param marker workspace used to keep track of column indices,
                  must be at least the size of the number of columns
            @retval the submatrix
        */
        SparseMatrix<T> GetSubMatrix(const std::vector<int>& rows,
                                     const std::vector<int>& cols,
                                     std::vector<int>& marker) const;

        /*! @brief Multiply by a scalar */
        template <typename T2 = T>
        SparseMatrix<T>& operator*=(T2 val);

        /*! @brief Divide by a scalar */
        template <typename T2 = T>
        SparseMatrix<T>& operator/=(T2 val);

        /*! @brief Set all nonzeros to a scalar */
        template <typename T2 = T>
        SparseMatrix<T>& operator=(T2 val);

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T2 = T>
        auto operator*(const VectorView<T2>& input) const;

        /*! @brief Sum of all data
            @retval sum Sum of all data
        */
        T Sum() const;

        /*! @brief Scale rows by given values
            @param values scale per row
        */
        void ScaleRows(const std::vector<T>& values);

        /*! @brief Scale cols by given values
            @param values scale per cols
        */
        void ScaleCols(const std::vector<T>& values);

        /*! @brief Permute the columns
            @param perm permutation to apply
        */
        void PermuteCols(const std::vector<int>& perm);

        /// Operator Requirement, calls the templated Mult
        void Mult(const VectorView<double>& input, VectorView<double>& output) const override;
        /// Operator Requirement, calls the templated MultAT
        void MultAT(const VectorView<double>& input, VectorView<double>& output) const override;

    private:
        size_t rows_;
        size_t cols_;
        size_t nnz_;

        std::vector<int> indptr_;
        std::vector<int> indices_;
        std::vector<T> data_;
};

template <typename T>
SparseMatrix<T>::SparseMatrix()
    : rows_(0), cols_(0), nnz_(0),
      indptr_(std::vector<int>(1, 0)), indices_(0), data_(0)
{

}

template <typename T>
SparseMatrix<T>::SparseMatrix(int size)
    : SparseMatrix<T>(size, size)
{

}

template <typename T>
SparseMatrix<T>::SparseMatrix(int rows, int cols)
    : rows_(rows), cols_(cols), nnz_(0),
      indptr_(std::vector<int>(rows + 1, 0)), indices_(0), data_(0)
{
    assert(rows_ >= 0);
    assert(cols_ >= 0);
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<int>& indptr,
                              const std::vector<int>& indices,
                              const std::vector<T>& data,
                              int rows, int cols)
    : rows_(rows), cols_(cols), nnz_(data.size()),
      indptr_(indptr), indices_(indices), data_(data)
{
    assert(rows_ >= 0);
    assert(cols_ >= 0);

    assert(indptr_.size() == rows_ + 1u);
    assert(indices_.size() == data_.size());
    assert(indptr_[0] == 0);
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<T>& diag)
    : rows_(diag.size()), cols_(diag.size()), nnz_(diag.size()),
      indptr_(diag.size() + 1), indices_(diag.size()), data_(diag)
{
    std::iota(begin(indptr_), end(indptr_), 0);
    std::iota(begin(indices_), end(indices_), 0);
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T>& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
      indptr_(other.indptr_), indices_(other.indices_), data_(other.data_)
{

}


template <typename T>
SparseMatrix<T>::SparseMatrix(SparseMatrix<T>&& other) noexcept
{
    Swap(*this, other);
}

template <typename T>
SparseMatrix<T>& SparseMatrix<T>::operator=(SparseMatrix<T> other) noexcept
{
    Swap(*this, other);

    return *this;
}

template <typename T>
void Swap(SparseMatrix<T>& lhs, SparseMatrix<T>& rhs)
{
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.cols_, rhs.cols_);
    std::swap(lhs.nnz_, rhs.nnz_);
    std::swap(lhs.indptr_, rhs.indptr_);
    std::swap(lhs.indices_, rhs.indices_);
    std::swap(lhs.data_, rhs.data_);
}

template <typename T>
void SparseMatrix<T>::Print(const std::string& label, std::ostream& out) const
{
    constexpr int width = 6;

    out << label << "\n";

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            out << "(" << i << ", " << indices_[j] << ") "
                << std::setw(width) << data_[j] << "\n";
        }
    }

    out << "\n";
}

template <typename T>
void SparseMatrix<T>::PrintDense(const std::string& label, std::ostream& out) const
{
    const DenseMatrix dense = ToDense();

    dense.Print(label, out);
}

template <typename T>
DenseMatrix SparseMatrix<T>::ToDense() const
{
    DenseMatrix dense(rows_, cols_);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            dense(i, indices_[j]) = data_[j];
        }
    }

    return dense;
}

template <typename T>
void SparseMatrix<T>::SortIndices()
{
    const auto compare_cols = [&](int i, int j)
    {
        return indices_[i] < indices_[j];
    };

    std::vector<int> permutation(indices_.size());
    std::iota(begin(permutation), end(permutation), 0);

    for (size_t i = 0; i < rows_; ++i)
    {
        const int start = indptr_[i];
        const int end = indptr_[i + 1];

        std::sort(begin(permutation) + start,
                  begin(permutation) + end,
                  compare_cols);
    }

    std::vector<int> sorted_indices(indices_.size());
    std::vector<T> sorted_data(data_.size());

    std::transform(begin(permutation), end(permutation), begin(sorted_indices),
                   [&] (int i)
    {
        return indices_[i];
    });
    std::transform(begin(permutation), end(permutation), begin(sorted_data),
                   [&] (int i)
    {
        return data_[i];
    });

    std::swap(indices_, sorted_indices);
    std::swap(data_, sorted_data);
}

template <typename T>
DenseMatrix SparseMatrix<T>::Mult(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.Cols());
    Mult(input, output);

    return output;
}

template <typename T>
DenseMatrix SparseMatrix<T>::MultAT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.Cols());
    MultAT(input, output);

    return output;
}

template <typename T>
void SparseMatrix<T>::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(input.Rows() == cols_);
    assert(output.Rows() == rows_);
    assert(output.Cols() == input.Cols());

    output = 0.0;

    for (size_t k = 0; k < input.Cols(); ++k)
    {
        for (size_t i = 0; i < rows_; ++i)
        {
            double val = 0.0;

            for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
            {
                val += data_[j] * input(indices_[j], k);
            }

            output(i, k) = val;
        }
    }
}

template <typename T>
void SparseMatrix<T>::MultAT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(input.Rows() == cols_);
    assert(output.Rows() == rows_);
    assert(output.Cols() == input.Cols());

    output = 0.0;

    for (size_t k = 0; k < input.Cols(); ++k)
    {
        for (size_t i = 0; i < rows_; ++i)
        {
            for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
            {
                output(indices_[j], k) += data_[j] * input(i, k);
            }
        }
    }
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::Transpose() const
{
    std::vector<int> out_indptr(cols_ + 1, 0);
    std::vector<int> out_indices(nnz_);
    std::vector<T> out_data(nnz_);

    for (const int& col : indices_)
    {
        out_indptr[col + 1]++;
    }

    for (size_t i = 0; i < cols_; ++i)
    {
        out_indptr[i + 1] += out_indptr[i];
    }

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            const int row = indices_[j];
            const T val = data_[j];

            out_indices[out_indptr[row]] = i;
            out_data[out_indptr[row]] = val;
            out_indptr[row]++;
        }
    }

    for (int i = cols_; i > 0; --i)
    {
        out_indptr[i] = out_indptr[i - 1];
    }

    out_indptr[0] = 0;

    return SparseMatrix(out_indptr, out_indices, out_data,
                        cols_, rows_);
}

template <typename T>
std::vector<double> SparseMatrix<T>::GetDiag() const
{
    assert(rows_ == cols_);

    std::vector<double> diag(rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        double val = 0.0;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (static_cast<size_t>(indices_[j]) == i)
            {
                val = data_[j];
            }
        }

        diag[i] = val;
    }

    return diag;
}

template <typename T>
void SparseMatrix<T>::AddDiag(const std::vector<T>& diag)
{
    assert(rows_ == cols_);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (static_cast<size_t>(indices_[j]) == i)
            {
                data_[j] += diag[i];
            }
        }
    }
}

template <typename T>
void SparseMatrix<T>::AddDiag(T val)
{
    assert(rows_ == cols_);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (static_cast<size_t>(indices_[j]) == i)
            {
                data_[j] += val;
            }
        }
    }
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::GetSubMatrix(const std::vector<int>& rows,
                                              const std::vector<int>& cols) const
{
    std::vector<int> marker(cols_, -1);

    return GetSubMatrix(rows, cols, marker);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::GetSubMatrix(const std::vector<int>& rows,
                                              const std::vector<int>& cols,
                                              std::vector<int>& marker) const
{
    assert(marker.size() >= cols_);

    std::vector<int> out_indptr(rows.size() + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    const size_t out_rows = rows.size();
    const size_t out_cols = cols.size();

    for (size_t i = 0; i < out_cols; ++i)
    {
        marker[cols[i]] = i;
    }

    for (size_t i = 0; i < out_rows; ++i)
    {
        const int row = rows[i];

        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (marker[indices_[j]] != -1)
            {
                ++out_nnz;
            }
        }

        out_indptr[i + 1] = out_nnz;
    }

    std::vector<int> out_indices(out_nnz);
    std::vector<T> out_data(out_nnz);

    int total = 0;

    for (auto row : rows)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (marker[indices_[j]] != -1)
            {
                out_indices[total] = marker[indices_[j]];
                out_data[total] = data_[j];

                total++;
            }
        }
    }

    for (auto i : cols)
    {
        marker[i] = -1;
    }

    return SparseMatrix<T>(out_indptr, out_indices, out_data,
                           out_rows, out_cols);
}

template <typename T>
inline
size_t SparseMatrix<T>::Rows() const
{
    return rows_;
}

template <typename T>
inline
size_t SparseMatrix<T>::Cols() const
{
    return cols_;
}

template <typename T>
inline
size_t SparseMatrix<T>::nnz() const
{
    return nnz_;
}

template <typename T>
inline
const std::vector<int>& SparseMatrix<T>::GetIndptr() const
{
    return indptr_;
}

template <typename T>
inline
const std::vector<int>& SparseMatrix<T>::GetIndices() const
{
    return indices_;
}

template <typename T>
inline
const std::vector<T>& SparseMatrix<T>::GetData() const
{
    return data_;
}

template <typename T>
inline
std::vector<int>& SparseMatrix<T>::GetIndptr()
{
    return indptr_;
}

template <typename T>
inline
std::vector<int>& SparseMatrix<T>::GetIndices()
{
    return indices_;
}

template <typename T>
inline
std::vector<T>& SparseMatrix<T>::GetData()
{
    return data_;
}

template <typename T>
inline
std::vector<int> SparseMatrix<T>::GetIndices(size_t row) const
{
    assert(row >= 0 && row < rows_);

    const size_t start = indptr_[row];
    const size_t end = indptr_[row + 1];

    return std::vector<int>(begin(indices_) + start, begin(indices_) + end);
}

template <typename T>
inline
std::vector<T> SparseMatrix<T>::GetData(size_t row) const
{
    assert(row >= 0 && row < rows_);

    const size_t start = indptr_[row];
    const size_t end = indptr_[row + 1];

    return std::vector<T>(begin(data_) + start, begin(data_) + end);
}

template <typename T>
template <typename T2>
auto SparseMatrix<T>::Mult(const VectorView<T2>& input) const
{
    Vector<typename std::common_type<T, T2>::type> output(rows_);
    Mult(input, output);

    return output;
}

template <typename T>
template <typename T2>
auto SparseMatrix<T>::MultAT(const VectorView<T2>& input) const
{
    Vector<typename std::common_type<T, T2>::type> output(cols_);
    MultAT(input, output);

    return output;
}

template <typename T>
template <typename T2, typename T3>
void SparseMatrix<T>::Mult(const VectorView<T2>& input, VectorView<T3>& output) const
{
    assert(input.size() == cols_);
    assert(output.size() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        T3 val = 0;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            val += data_[j] * input[indices_[j]];
        }

        output[i] = val;
    }
}

template <typename T>
template <typename T2, typename T3>
void SparseMatrix<T>::MultAT(const VectorView<T2>& input, VectorView<T3>& output) const
{
    assert(input.size() == rows_);
    assert(output.size() == cols_);

    std::fill(std::begin(output), std::end(output), 0.0);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            output[indices_[j]] += data_[j] * input[i];
        }
    }
}

template <typename T>
void SparseMatrix<T>::Mult(const VectorView<double>& input, VectorView<double>& output) const
{
    Mult<double, double>(input, output);
}

template <typename T>
void SparseMatrix<T>::MultAT(const VectorView<double>& input, VectorView<double>& output) const
{
    MultAT<double, double>(input, output);
}

template <typename T>
template <typename T2>
auto SparseMatrix<T>::operator*(const VectorView<T2>& input) const
{
    return Mult<T2>(input);
}

template <typename T>
template <typename T2, typename T3>
auto SparseMatrix<T>::Mult(const SparseMatrix<T2>& rhs) const
{
    std::vector<int> marker(rhs.Cols(), -1);

    std::vector<int> out_indptr(rows_ + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    const std::vector<int>& rhs_indptr = rhs.GetIndptr();
    const std::vector<int>& rhs_indices = rhs.GetIndices();
    const std::vector<T2>& rhs_data = rhs.GetData();

    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] != static_cast<int>(i))
                {
                    marker[rhs_indices[k]] = i;
                    ++out_nnz;
                }
            }
        }

        out_indptr[i + 1] = out_nnz;
    }

    std::fill(begin(marker), end(marker), -1);

    std::vector<int> out_indices(out_nnz);
    std::vector<T3> out_data(out_nnz);

    int total = 0;

    for (size_t i = 0; i < rows_; ++i)
    {
        int row_nnz = total;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] < row_nnz)
                {
                    marker[rhs_indices[k]] = total;
                    out_indices[total] = rhs_indices[k];
                    out_data[total] = data_[j] * rhs_data[k];

                    total++;
                }
                else
                {
                    out_data[marker[rhs_indices[k]]] += data_[j] * rhs_data[k];
                }
            }
        }
    }

    return SparseMatrix<T3>(out_indptr, out_indices, out_data,
                            rows_, rhs.Cols());
}

template <typename T>
template <typename T2>
SparseMatrix<T>& SparseMatrix<T>::operator*=(T2 val)
{
    for (auto& i : data_)
    {
        i *= val;
    }

    return *this;
}

template <typename T>
template <typename T2>
SparseMatrix<T>& SparseMatrix<T>::operator/=(T2 val)
{
    assert(val != 0);

    for (auto& i : data_)
    {
        i /= val;
    }

    return *this;
}

template <typename T>
template <typename T2>
SparseMatrix<T>& SparseMatrix<T>::operator=(T2 val)
{
    assert(val != 0);

    std::fill(begin(data_), end(data_), val);

    return *this;
}

template <typename T>
size_t SparseMatrix<T>::RowSize(size_t row) const
{
    assert(row >= 0 && row < rows_);

    return indptr_[row + 1] - indptr_[row];
}

/*! @brief Multiply a sparse matrix and
           a scalar into a new sparse matrix: Aa = B
    @param lhs the left hand side matrix A
    @param val the right hand side scalar a
    @retval the multiplied matrix B
*/
template <typename T2, typename T3>
SparseMatrix<T2> operator*(SparseMatrix<T2> lhs, T3 val)
{
    return lhs *= val;
}

/*! @brief Multiply a sparse matrix and
           a scalar into a new sparse matrix: aA = B
    @param val the left hand side scalar a
    @param rhs the right hand side matrix A
    @retval the multiplied matrix B
*/
template <typename T2, typename T3>
SparseMatrix<T2> operator*(T3 val, SparseMatrix<T2> rhs)
{
    return rhs *= val;
}

template <typename T>
T SparseMatrix<T>::Sum() const
{
    T sum = std::accumulate(std::begin(data_), std::end(data_), 0.0);
    return sum;
}

template <typename T>
void SparseMatrix<T>::ScaleRows(const std::vector<T>& values)
{
    for (size_t i = 0; i < rows_; ++i)
    {
        const double scale = values[i];

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            data_[j] *= scale;
        }
    }
}

template <typename T>
void SparseMatrix<T>::ScaleCols(const std::vector<T>& values)
{
    for (size_t i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            data_[j] *= values[indices_[j]];
        }
    }
}

template <typename T>
void SparseMatrix<T>::PermuteCols(const std::vector<int>& perm)
{
    assert(perm.size() == cols_);

    for (size_t i = 0; i < indices_.size(); ++i)
    {
        assert(perm[indices_[i]] >= 0);
        assert(perm[indices_[i]] < static_cast<int>(cols_));

        indices_[i] = perm[indices_[i]];
    }
}

} //namespace linalgcpp

#endif // SPARSEMATRIX_HPP__
