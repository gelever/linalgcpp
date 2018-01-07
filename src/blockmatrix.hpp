/*! @file */

#ifndef BLOCKMATRIX_HPP__
#define BLOCKMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

#include "operator.hpp"
#include "sparsematrix.hpp"

namespace linalgcpp
{

/*! @brief Dense block matrix where each block is a sparse matrix
           Blocks of zero size are assumed to be zero matrix
           Offsets are to be of size (blocks + 1), where the last 
           entry is the total size.

           @warning This is NOT a view.  BlockMatrix copies/takes
           possession of all given input!
*/
template <typename T = double>
class BlockMatrix : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        BlockMatrix();

        /*! @brief Square Constructor with given symmetric offsets*/
        explicit BlockMatrix(std::vector<size_t> offsets);

        /*! @brief Rectangle Constructor with given offsets*/
        BlockMatrix(std::vector<size_t> row_offsets, std::vector<size_t> col_offsets);

        /*! @brief Default deconstructor */
        ~BlockMatrix() noexcept = default;
        /*! @brief The number of rows in this matrix
            @retval the number of rows
        */
        size_t Rows() const override;

        /*! @brief The number of columns in this matrix
            @retval the number of columns
        */
        size_t Cols() const override;

        /*! @brief Get the row offsets
            @retval the row offsets
        */
        const std::vector<size_t>& GetRowOffsets() const;

        /*! @brief Get the col offsets
            @retval the col offsets
        */
        const std::vector<size_t>& GetColOffsets() const;

        /*! @brief Get a block
            @param i row index
            @param j column index
            @retval SparseMatrix at index (i, j)
        */
        const SparseMatrix<T>& GetBlock(size_t i, size_t j) const;

        /*! @brief Set a block
            @param i row index
            @param j column index
            @param SparseMatrix at index (i, j)
        */
        void SetBlock(size_t i, size_t j, SparseMatrix<T> mat);

        /*! @brief Get the transpose matrix
        */
        BlockMatrix<T> Transpose() const;

        /*! @brief Get the combined monolithic matrix
        */
        SparseMatrix<T> Combine() const;

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

        /// Operator Requirement
        void Mult(const VectorView<double>& input, VectorView<double>& output) const override;
        /// Operator Requirement
        void MultAT(const VectorView<double>& input, VectorView<double>& output) const override;

        using Operator::Mult;

    private:
        std::vector<size_t> row_offsets_;
        std::vector<size_t> col_offsets_;

        std::vector<std::vector<SparseMatrix<T>>> A_;

        size_t rows_;
        size_t cols_;
};

template <typename T>
BlockMatrix<T>::BlockMatrix()
    : row_offsets_(1, 0), col_offsets_(1, 0), rows_(0), cols_(0)
{

}

template <typename T>
BlockMatrix<T>::BlockMatrix(std::vector<size_t> offsets) :
    row_offsets_(offsets), col_offsets_(std::move(offsets)),
    A_(row_offsets_.size() - 1, std::vector<SparseMatrix<T>>(col_offsets_.size() - 1)),
    rows_(row_offsets_.back()), cols_(col_offsets_.back())
{

}

template <typename T>
BlockMatrix<T>::BlockMatrix(std::vector<size_t> row_offsets, std::vector<size_t> col_offsets)
    : row_offsets_(std::move(row_offsets)), col_offsets_(std::move(col_offsets)),
    A_(row_offsets_.size() - 1, std::vector<SparseMatrix<T>>(col_offsets_.size() - 1)),
    rows_(row_offsets_.back()), cols_(col_offsets_.back())
{

}

template <typename T>
size_t BlockMatrix<T>::Rows() const
{
    return rows_;
}

template <typename T>
size_t BlockMatrix<T>::Cols() const
{
    return cols_;
}

template <typename T>
const std::vector<size_t>& BlockMatrix<T>::GetRowOffsets() const
{
    return row_offsets_;
}

template <typename T>
const std::vector<size_t>& BlockMatrix<T>::GetColOffsets() const
{
    return col_offsets_;
}

template <typename T>
const SparseMatrix<T>& BlockMatrix<T>::GetBlock(size_t i, size_t j) const
{
    assert(i < row_offsets_.size() - 1);
    assert(j < col_offsets_.size() - 1);

    return A_[i][j];
}

template <typename T>
void BlockMatrix<T>::SetBlock(size_t i, size_t j, SparseMatrix<T> mat)
{
    assert(i < row_offsets_.size() - 1);
    assert(j < col_offsets_.size() - 1);
    assert(mat.Rows() == (row_offsets_[i + 1] - row_offsets_[i]));
    assert(mat.Cols() == (col_offsets_[i + 1] - col_offsets_[i]));

    Swap(A_[i][j], mat);
}

template <typename T>
BlockMatrix<T> BlockMatrix<T>::Transpose() const
{
    BlockMatrix<T> transpose(col_offsets_, row_offsets_);

    const size_t row_blocks = row_offsets_.size() - 1;
    const size_t col_blocks = col_offsets_.size() - 1;

    for (size_t i = 0; i < row_blocks; ++i)
    {
        for (size_t j = 0; j < col_blocks; ++j)
        {
            transpose.SetBlock(j, i, GetBlock(i, j));
        }
    }

    return transpose;
}

template <typename T>
SparseMatrix<T> BlockMatrix<T>::Combine() const
{
    size_t nnz = 0;
    for (const auto& row : A_)
    {
        for (const auto& mat : row)
        {
            nnz += mat.nnz();
        }
    }

    std::vector<int> indptr(rows_ + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    indptr[0] = 0;
    
    size_t nnz_count = 0;

    for (size_t i = 0; i < row_offsets_.size() - 1; ++i)
    {
        size_t local_row = 0;

        for (size_t row = row_offsets_[i]; row < row_offsets_[i + 1]; ++row)
        {
            for (size_t j = 0; j < col_offsets_.size() - 1; ++j)
            {
                const auto& mat = A_[i][j];
                if (mat.Rows() == 0 || mat.Cols() == 0)
                {
                    continue;
                }

                const auto offset = col_offsets_[j]; 

                const auto& mat_indptr = mat.GetIndptr();
                const auto& mat_indices = mat.GetIndices();
                const auto& mat_data = mat.GetData();

                const int start = mat_indptr[local_row];
                const int end = mat_indptr[local_row + 1];

                for (int jj = start; jj < end; ++jj)
                {
                    indices[nnz_count] = offset + mat_indices[jj];
                    data[nnz_count] = mat_data[jj];

                    nnz_count++;
                }
            }

            local_row++;

            indptr[row + 1] = nnz_count;
        }
    }

    return SparseMatrix<T>(indptr, indices, data, rows_, cols_);
}

template <typename T>
void BlockMatrix<T>::Print(const std::string& label, std::ostream& out) const
{
    out << label << "\n";

    for (size_t i = 0; i < row_offsets_.size() - 1; ++i)
    {
        for (size_t j = 0; j < col_offsets_.size() - 1; ++j)
        {
            const auto& mat = A_[i][j];

            std::stringstream block_label;
            block_label << "block (" << i << "," << j << ")";

            mat.Print(block_label.str(), out);
        }
    }
}

template <typename T>
void BlockMatrix<T>::PrintDense(const std::string& label, std::ostream& out) const
{
    Combine().PrintDense(label, out);
}

template <typename T>
void BlockMatrix<T>::Mult(const VectorView<double>& input, VectorView<double>& output) const
{
    assert(input.size() == cols_);
    assert(output.size() == rows_);

    output = 0.0;

    for (size_t i = 0; i < row_offsets_.size() - 1; ++i)
    {
        size_t local_row = 0;

        for (size_t row = row_offsets_[i]; row < row_offsets_[i + 1]; ++row)
        {
            double val = 0.0;

            for (size_t j = 0; j < col_offsets_.size() - 1; ++j)
            {
                const auto& mat = A_[i][j];

                if (mat.Rows() == 0 || mat.Cols() == 0)
                {
                    continue;
                }

                const auto offset = col_offsets_[j]; 

                const auto& mat_indptr = mat.GetIndptr();
                const auto& mat_indices = mat.GetIndices();
                const auto& mat_data = mat.GetData();

                const int start = mat_indptr[local_row];
                const int end = mat_indptr[local_row + 1];

                for (int jj = start; jj < end; ++jj)
                {
                    const size_t col = offset + mat_indices[jj];

                    val += mat_data[jj] * input[col];

                }
            }

            local_row++;

            output[row] = val;
        }
    }
}

template <typename T>
void BlockMatrix<T>::MultAT(const VectorView<double>& input, VectorView<double>& output) const
{
    assert(input.size() == rows_);
    assert(output.size() == cols_);

    output = 0.0;

    for (size_t i = 0; i < row_offsets_.size() - 1; ++i)
    {
        size_t local_row = 0;

        for (size_t row = row_offsets_[i]; row < row_offsets_[i + 1]; ++row)
        {
            for (size_t j = 0; j < col_offsets_.size() - 1; ++j)
            {
                const auto& mat = A_[i][j];

                if (mat.Rows() == 0 || mat.Cols() == 0)
                {
                    continue;
                }

                const auto offset = col_offsets_[j]; 

                const auto& mat_indptr = mat.GetIndptr();
                const auto& mat_indices = mat.GetIndices();
                const auto& mat_data = mat.GetData();

                const int start = mat_indptr[local_row];
                const int end = mat_indptr[local_row + 1];

                for (int jj = start; jj < end; ++jj)
                {
                    const size_t col = offset + mat_indices[jj];

                    output[col] += mat_data[jj] * input[row];

                }
            }

            local_row++;
        }
    }
}


} //namespace linalgcpp

#endif // BLOCKMATRIX_HPP__
