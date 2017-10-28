/*! @file */

#ifndef COOMATRIX_HPP__
#define COOMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <assert.h>

#include "sparsematrix.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

/*! @brief Coordinate Matrix that keeps track
           of individual entries in a matrix.

    @note Multiple entries for a single coordinate
    are summed together
*/
template <typename T>
class CooMatrix : public Operator
{
    public:
        /*! @brief Default Constructor
            The size of the matrix is determined
            by the largest entry.
        */
        CooMatrix();

        /*! @brief Square Constructor
            @param size the number of rows and columns
        */
        CooMatrix(int size);

        /*! @brief Rectangle Constructor
            @param rows the number of rows
            @param cols the number of columns
        */
        CooMatrix(int rows, int cols);

        /*! @brief Copy Constructor */
        CooMatrix(const CooMatrix& other) = default;

        /*! @brief Destructor */
        ~CooMatrix() noexcept = default;

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
        void Mult(const Vector<double>& input, Vector<double>& output) const override;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        void MultAT(const Vector<double>& input, Vector<double>& output) const override;

        /*! @brief Gets the value at a point
            @param i row index
            @param j column index

            @throws if value is not in matrix
        */
        const T& operator()(int i, int j) const;

        /*! @brief Print all entries
            @param label label to print before data
        */
        void Print(const std::string& label = "") const;

    private:
        int rows_;
        int cols_;

        bool size_set_;

        std::tuple<size_t, size_t> FindSize() const;

        struct tuple_hash
        {
            std::size_t operator()(const std::tuple<int, int>& tup) const noexcept
            {
                std::size_t i = std::hash<int>{}(std::get<0>(tup));
                std::size_t j = std::hash<int>{}(std::get<1>(tup));

                return i ^ (j << 1);
            }
        };

        std::unordered_map<std::tuple<int, int>, T, tuple_hash> entries_map;
};

template <typename T>
CooMatrix<T>::CooMatrix()
    : rows_(0), cols_(0), size_set_(false)
{
}

template <typename T>
CooMatrix<T>::CooMatrix(int size) : CooMatrix(size, size)
{

}

template <typename T>
CooMatrix<T>::CooMatrix(int rows, int cols)
    : rows_(rows), cols_(cols), size_set_(true)
{

}

template <typename T>
size_t CooMatrix<T>::Rows() const
{
    size_t rows;
    std::tie(rows, std::ignore) = FindSize();

    return rows;
}

template <typename T>
size_t CooMatrix<T>::Cols() const
{
    size_t cols;
    std::tie(std::ignore, cols) = FindSize();

    return cols;
}

template <typename T>
void CooMatrix<T>::SetSize(size_t rows, size_t cols)
{
    size_set_ = true;

    rows_ = rows;
    cols_ = cols;
}

template <typename T>
void CooMatrix<T>::Add(int i, int j, T val)
{
    assert(i >= 0);
    assert(j >= 0);
    assert(val == val); // is finite

    if (size_set_)
    {
        assert(i < rows_);
        assert(j < cols_);
    }

    entries_map[std::make_tuple(i,j)] += val;
}

template <typename T>
void CooMatrix<T>::AddSym(int i, int j, T val)
{
    Add(i, j, val);

    if (i != j)
    {
        Add(j, i, val);
    }
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& indices,
                       const DenseMatrix& values)
{
    Add(indices, indices, values);
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& rows,
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
DenseMatrix CooMatrix<T>::ToDense() const
{
    if (entries_map.size() == 0)
    {
        return DenseMatrix();
    }

    size_t rows;
    size_t cols;
    std::tie(rows, cols) = FindSize();

    DenseMatrix dense(rows, cols);

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;

        int i = std::get<0>(tup);
        int j = std::get<1>(tup);
        double val = entry.second;

        dense(i, j) += val;
    }

    return dense;
}

template <typename T>
template <typename T2>
SparseMatrix<T2> CooMatrix<T>::ToSparse() const
{
    if (entries_map.size() == 0)
    {
        return SparseMatrix<T2>();
    }

    size_t rows;
    size_t cols;
    std::tie(rows, cols) = FindSize();

    std::vector<int> indptr(rows + 1, 0);
    std::vector<int> indices(entries_map.size());
    std::vector<T2> data(entries_map.size());

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;
        const int i = std::get<0>(tup);

        indptr[i + 1]++;
    }

    for (size_t i = 0; i < rows; ++i)
    {
        indptr[i + 1] += indptr[i];
    }

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;
        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const T val = entry.second;

        indices[indptr[i]] = j;
        data[indptr[i]] = val;
        indptr[i]++;
    }

    for (int i = rows; i > 0; --i)
    {
        indptr[i] = indptr[i - 1];
    }

    indptr[0] = 0;

    return SparseMatrix<T2>(indptr, indices, data, rows, cols);
}


template <typename T>
void CooMatrix<T>::Mult(const Vector<double>& input, Vector<double>& output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;

        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const double val = entry.second;

        output[i] += val * input[j];
    }
}

template <typename T>
void CooMatrix<T>::MultAT(const Vector<double>& input, Vector<double>& output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;

        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const double val = entry.second;

        output[j] += val * input[i];
    }
}

template <typename T>
std::tuple<size_t, size_t> CooMatrix<T>::FindSize() const
{
    if (size_set_)
    {
        return std::make_tuple<size_t, size_t>(rows_, cols_);
    }

    int rows = 0;
    int cols = 0;

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;

        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);

        rows = std::max(rows, i);
        cols = std::max(cols, j);
    }

    return std::make_tuple<size_t, size_t>(rows + 1, cols + 1);
}

template <typename T>
const T& CooMatrix<T>::operator()(int i, int j) const
{
    auto tup = std::make_tuple<int, int>(std::move(i), std::move(j));
    auto search = entries_map.find(tup);

    if (search != entries_map.end())
    {
        return search->second;
    }
    else
    {
        throw std::runtime_error("Entry not found in Coo Matrix!\n");
    }

}

template <typename T>
void CooMatrix<T>::Print(const std::string& label) const
{
    std::cout << label << "\n";

    for (const auto& entry : entries_map)
    {
        const auto& tup = entry.first;

        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const T val = entry.second;

        std::cout << "(" << i << ", " << j << ") " << val << "\n";
    }

    std::cout << "\n";

}

} // namespace linalgcpp

#endif // COOMATRIX_HPP__
