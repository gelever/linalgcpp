#ifndef COOMATRIX_HPP__
#define COOMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <queue>
#include <tuple>
#include <assert.h>

#include "sparsematrix.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

template <typename T>
class CooMatrix : public Operator
{
    public:
        CooMatrix();
        CooMatrix(int size);
        CooMatrix(int rows, int cols);

        CooMatrix(const CooMatrix& other) = default;
        ~CooMatrix() noexcept = default;

        size_t Rows() const;
        size_t Cols() const;

        void Add(int i, int j, T val);
        void AddSym(int i, int j, T val);
        void Add(const std::vector<int>& rows,
                 const std::vector<int>& cols,
                 const DenseMatrix& values);

        template <typename T2 = T>
        SparseMatrix<T2> ToSparse() const;

        DenseMatrix ToDense() const;

        void Mult(const Vector<double>& input, Vector<double>& output) const override;
        void MultAT(const Vector<double>& input, Vector<double>& output) const override;

    private:
        int rows_;
        int cols_;

        mutable std::vector<std::tuple<int, int, T>> entries;
};

template <typename T>
CooMatrix<T>::CooMatrix()
    : rows_(-1), cols_(-1)
{
}

template <typename T>
CooMatrix<T>::CooMatrix(int size) : CooMatrix(size, size)
{

}

template <typename T>
CooMatrix<T>::CooMatrix(int rows, int cols)
    : rows_(rows), cols_(cols)
{

}

template <typename T>
size_t CooMatrix<T>::Rows() const
{
    if (rows_ > -1)
    {
        return rows_;
    }

    if (entries.size() == 0)
    {
        return 0;
    }

    auto max_el = *std::max_element(begin(entries), end(entries));
    return std::get<0>(max_el) + 1;
}

template <typename T>
size_t CooMatrix<T>::Cols() const
{
    if (cols_ > -1)
    {
        return cols_;
    }

    if (entries.size() == 0)
    {
        return 0;
    }

    auto max_el = *std::max_element(begin(entries), end(entries));
    return std::get<1>(max_el) + 1;
}

template <typename T>
void CooMatrix<T>::Add(int i, int j, T val)
{
    assert(i >= 0);
    assert(j >= 0);
    assert(val == val); // is finite

    if (rows_ > -1)
    {
        assert(i < rows_);
        assert(j < cols_);
    }

    entries.push_back(std::make_tuple(i, j, val));
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
    if (entries.size() == 0)
    {
        return DenseMatrix();
    }

    int rows;
    int cols;

    if (rows_ > -1)
    {
        rows = rows_;
        cols = cols_;
    }
    else
    {
        auto max_el = *std::max_element(begin(entries), end(entries));
        rows = std::get<0>(max_el) + 1;
        cols = std::get<1>(max_el) + 1;
    }

    DenseMatrix dense(rows, cols);

    for (const auto& tup : entries)
    {
        int i = std::get<0>(tup);
        int j = std::get<1>(tup);
        double val = std::get<2>(tup);

        dense(i, j) += val;
    }

    return dense;
}

template <typename T>
template <typename T2>
SparseMatrix<T2> CooMatrix<T>::ToSparse() const
{
    if (entries.size() == 0)
    {
        return SparseMatrix<T2>();
    }

    std::sort(begin(entries), end(entries));

    int rows;
    int cols;

    if (rows_ > -1)
    {
        rows = rows_;
        cols = cols_;
    }
    else
    {
        auto max_el = entries.back();
        rows = std::get<0>(max_el) + 1;
        cols = std::get<1>(max_el) + 1;
    }

    assert(rows >= 0);
    assert(cols >= 0);

    std::vector<int> indptr(rows + 1);
    std::vector<int> indices;
    std::vector<T2> data;

    indptr[0] = 0;

    int current_row = 0;

    for (const auto& tup : entries)
    {
        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const T val = std::get<2>(tup);

        // Set Indptr if at new row
        if (i != current_row)
        {
            for (int ii = current_row; ii < i; ++ii)
            {
                indptr[ii + 1] = data.size();
            }
        }

        // Add data and indices
        if (indices.size() && j == indices.back() && i == current_row)
        {
            data.back() += val;
        }
        else
        {
            indices.push_back(j);
            data.push_back(val);
        }

        current_row = i;
    }

    std::fill(begin(indptr) + current_row + 1,
              end(indptr), data.size());

    return SparseMatrix<T2>(indptr, indices, data, rows, cols);
}

template <typename T>
void CooMatrix<T>::Mult(const Vector<double>& input, Vector<double>& output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    for (const auto& tup : entries)
    {
        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const T val = std::get<2>(tup);

        output[i] += val * input[j];
    }
}

template <typename T>
void CooMatrix<T>::MultAT(const Vector<double>& input, Vector<double>& output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    for (const auto& tup : entries)
    {
        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const T val = std::get<2>(tup);

        output[j] += val * input[i];
    }
}

} // namespace linalgcpp

#endif // COOMATRIX_HPP__
