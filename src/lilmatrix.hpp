/*! @file */

#ifndef LILMATRIX_HPP__
#define LILMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <queue>
#include <tuple>
#include <list>
#include <assert.h>

#include "sparsematrix.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

/*! @brief Linked List Matrix that keeps track
           of individual entries in a matrix.

    @note Multiple entries for a single coordinate
    are summed together
*/
template <typename T>
class LilMatrix : public Operator
{
    public:
        /*! @brief Default Constructor
            The size of the matrix is determined
            by the largest entry.
        */
        LilMatrix();

        /*! @brief Square Constructor
            @param size the number of rows and columns
        */
        explicit LilMatrix(int size);

        /*! @brief Rectangle Constructor
            @param rows the number of rows
            @param cols the number of columns
        */
        LilMatrix(int rows, int cols);

        /*! @brief Copy Constructor */
        LilMatrix(const LilMatrix& other) noexcept;

        /*! @brief Move Constructor */
        LilMatrix(LilMatrix&& other) noexcept;

        /*! @brief Assignment Operator */
        LilMatrix& operator=(LilMatrix other) noexcept;

        /*! @brief Destructor */
        ~LilMatrix() noexcept = default;

        /*! @brief Swap two matrices
            @param lhs left hand side matrix
            @param rhs right hand side matrix
        */
        template <typename T2>
        friend void Swap(LilMatrix<T2>& lhs, LilMatrix<T2>& rhs);

        /*! @brief Get the number of rows.
            @retval the number of rows

            @note If the matrix size was not specified during
            creation, then the number of rows is determined
            by the maximum element.
        */
        size_t Rows() const override;

        /*! @brief Get the number of columns.
            @retval the number of columns

            @note If the matrix size was not specified during
            creation, then the number of columns is determined
            by the maximum element
        */
        size_t Cols() const override;

        /*! @brief Set the size of the matrix
            @param rows the number of rows
            @param cols the number of columns
        */
        void SetSize(size_t rows, size_t cols);

        /*! @brief Add an entry to the matrix
            @param i row index
            @param j column index
            @param val value to add
        */
        void Add(int i, int j, T val);

        /*! @brief Add an entry to the matrix and its symmetric counterpart
            @param i row index
            @param j column index
            @param val value to add
        */
        void AddSym(int i, int j, T val);

        /*! @brief Add a dense matrix worth of entries
            @param indices row and column indices to add
            @param values the values to add
        */

        void Add(const std::vector<int>& indices,
                 const DenseMatrix& values);

        /*! @brief Add a dense matrix worth of entries
            @param rows set of row indices
            @param cols set of column indices
            @param values the values to add
        */
        void Add(const std::vector<int>& rows,
                 const std::vector<int>& cols,
                 const DenseMatrix& values);

        /*! @brief Generate a sparse matrix from the entries
            @retval SparseMatrix containing all the entries

            @note Multiple entries for a single coordinate
            are summed together
        */
        template <typename T2 = T>
        SparseMatrix<T2> ToSparse() const;

        /*! @brief Generate a dense matrix from the entries
            @retval DenseMatrix containing all the entries

            @note Multiple entries for a single coordinate
            are summed together
        */
        DenseMatrix ToDense() const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        void Mult(const VectorView<double>& input, VectorView<double>& output) const override;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        void MultAT(const VectorView<double>& input, VectorView<double>& output) const override;


        /*! @brief Print all entries
            @param label label to print before data
            @param out stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Eliminate entries with zero value
            @param tolerance how small of values to erase
        */
        void EliminateZeros(double tolerance = 0);

    private:
        std::tuple<size_t, size_t> FindSize() const;

        size_t rows_;
        size_t cols_;

        bool size_set_;

        mutable std::vector<std::list<std::pair<int, T>>> entries_;
};

template <typename T>
LilMatrix<T>::LilMatrix()
    : rows_(0), cols_(0), size_set_(false)
{
}

template <typename T>
LilMatrix<T>::LilMatrix(int size) : LilMatrix(size, size)
{

}

template <typename T>
LilMatrix<T>::LilMatrix(int rows, int cols)
    : rows_(rows), cols_(cols), size_set_(true), entries_(rows_)
{
    assert(rows >= 0);
    assert(cols >= 0);
}

template <typename T>
LilMatrix<T>::LilMatrix(const LilMatrix& other) noexcept
    : rows_(other.rows_), cols_(other.cols_),
      size_set_(other.size_set_), entries_(other.entries_)
{

}

template <typename T>
LilMatrix<T>::LilMatrix(LilMatrix&& other) noexcept
{
    Swap(*this, other);
}

template <typename T>
LilMatrix<T>& LilMatrix<T>::operator=(LilMatrix other) noexcept
{
    Swap(*this, other);

    return *this;
}

template <typename T>
void Swap(LilMatrix<T>& lhs, LilMatrix<T>& rhs)
{
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.cols_, rhs.cols_);
    std::swap(lhs.size_set_, rhs.size_set_);
    std::swap(lhs.entries_, rhs.entries_);
}

template <typename T>
size_t LilMatrix<T>::Rows() const
{
    size_t rows;
    std::tie(rows, std::ignore) = FindSize();

    return rows;
}

template <typename T>
size_t LilMatrix<T>::Cols() const
{
    size_t cols;
    std::tie(std::ignore, cols) = FindSize();

    return cols;
}

template <typename T>
void LilMatrix<T>::SetSize(size_t rows, size_t cols)
{
    size_set_ = true;

    rows_ = rows;
    cols_ = cols;
}

template <typename T>
void LilMatrix<T>::Add(int i, int j, T val)
{
    assert(i >= 0);
    assert(j >= 0);
    assert(val == val); // is finite

    if (size_set_)
    {
        assert(static_cast<size_t>(i) < rows_);
        assert(static_cast<size_t>(j) < cols_);
    }
    else if (static_cast<size_t>(i) >= entries_.size())
    {
        entries_.resize(i + 1);
    }

    entries_[i].emplace_front(j, val);
}

template <typename T>
void LilMatrix<T>::AddSym(int i, int j, T val)
{
    Add(i, j, val);

    if (i != j)
    {
        Add(j, i, val);
    }
}

template <typename T>
void LilMatrix<T>::Add(const std::vector<int>& indices,
                       const DenseMatrix& values)
{
    Add(indices, indices, values);
}

template <typename T>
void LilMatrix<T>::Add(const std::vector<int>& rows,
                       const std::vector<int>& cols,
                       const DenseMatrix& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.Rows()));
    assert(cols.size() == static_cast<unsigned int>(values.Cols()));

    const int num_rows = values.Rows();
    const int num_cols = values.Cols();

    for (int j = 0; j < num_cols; ++j)
    {
        const int col = cols[j];

        for (int i = 0; i < num_rows; ++i)
        {
            const int row = rows[i];
            const double val = values(i, j);

            Add(row, col, val);
        }
    }
}

template <typename T>
DenseMatrix LilMatrix<T>::ToDense() const
{
    if (entries_.size() == 0)
    {
        return DenseMatrix();
    }

    size_t rows;
    size_t cols;
    std::tie(rows, cols) = FindSize();

    DenseMatrix dense(rows, cols);

    const size_t size = entries_.size();

    for (size_t i = 0; i < size; ++i)
    {
        for (const auto& node : entries_[i])
        {
            dense(i, node.first) += node.second;
        }
    }

    return dense;
}

template <typename T>
template <typename T2>
SparseMatrix<T2> LilMatrix<T>::ToSparse() const
{
    size_t rows;
    size_t cols;
    std::tie(rows, cols) = FindSize();

    if (entries_.size() == 0)
    {
        return SparseMatrix<T2>(rows, cols);
    }

    const size_t size = entries_.size();

    for (auto& row : entries_)
    {
        row.sort();
    }

    const size_t nnz = entries_.size();

    std::vector<int> indptr(rows + 1, 0);
    std::vector<int> indices;
    std::vector<T2> data;

    indices.reserve(nnz);
    data.reserve(nnz);

    indptr[0] = 0;

    for (size_t i = 0; i < size; ++i)
    {
        const size_t current_row = static_cast<size_t>(indptr[i]);

        for (const auto& node : entries_[i])
        {
            const int& j = node.first;
            const T& val = node.second;

            if (indices.size() != current_row && j == indices.back())
            {
                data.back() += val;
            }
            else
            {
                indices.push_back(j);
                data.push_back(val);
            }
        }

        indptr[i + 1] = data.size();
    }

    return SparseMatrix<T2>(indptr, indices, data, rows, cols);
}

template <typename T>
void LilMatrix<T>::Mult(const VectorView<double>& input, VectorView<double>& output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    const size_t size = entries_.size();

    for (size_t i = 0; i < size; ++i)
    {
        const auto& list = entries_[i];

        for (const auto& node : list)
        {
            const int j = node.first;
            const T val = node.second;

            output[i] += val * input[j];
        }
    }
}

template <typename T>
void LilMatrix<T>::MultAT(const VectorView<double>& input, VectorView<double>& output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    const size_t size = entries_.size();

    for (size_t i = 0; i < size; ++i)
    {
        const auto& list = entries_[i];

        for (const auto& node : list)
        {
            const int j = node.first;
            const T val = node.second;

            output[j] += val * input[i];
        }
    }
}

template <typename T>
void LilMatrix<T>::Print(const std::string& label, std::ostream& out) const
{
    out << label << "\n";

    const size_t size = entries_.size();

    for (size_t i = 0; i < size; ++i)
    {
        const auto& list = entries_[i];

        for (const auto& node : list)
        {
            const int j = node.first;
            const T val = node.second;

            out << "(" << i << ", " << j << ") " << val << "\n";
        }
    }

    out << "\n";
}

template <typename T>
void LilMatrix<T>::EliminateZeros(double tolerance)
{
    for (auto& row : entries_)
    {
        row.erase(std::remove_if(std::begin(row), std::end(row),
                                 [&](const auto & entry)
        {
            return std::abs(entry.second) < tolerance;
        }),
        std::end(row));
    }
}

template <typename T>
std::tuple<size_t, size_t> LilMatrix<T>::FindSize() const
{
    if (size_set_)
    {
        return std::tuple<size_t, size_t> {rows_, cols_};
    }

    const size_t rows = entries_.size();
    int cols = 0;

    for (const auto& row : entries_)
    {
        for (const auto& node : row)
        {
            cols = std::max(cols, node.first);
        }
    }

    return std::tuple<size_t, size_t> {rows + 1, cols + 1};
}

} // namespace linalgcpp

#endif // LILMATRIX_HPP__
