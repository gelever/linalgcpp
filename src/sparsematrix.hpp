#ifndef SPARSEMATRIX_HPP__
#define SPARSEMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

#include "densematrix.hpp"

namespace linalgcpp
{

template <typename T = double>
class SparseMatrix
{
    public:
        SparseMatrix();
        SparseMatrix(const std::vector<int>& indptr,
                     const std::vector<int>& indices,
                     const std::vector<T>& data,
                     int rows, int cols);

        SparseMatrix(const std::vector<T>& diag);

        SparseMatrix(const SparseMatrix<T>& other);
        ~SparseMatrix() noexcept = default;

        SparseMatrix<T>& operator=(SparseMatrix<T> other);
        SparseMatrix(SparseMatrix<T>&& other);

        template <typename T2>
        friend void Swap(SparseMatrix<T2>& lhs, SparseMatrix<T2>& rhs);

        int Rows() const;
        int Cols() const;
        int nnz() const;

        const std::vector<int>& GetIndptr() const;
        const std::vector<int>& GetIndices() const;
        const std::vector<T>& GetData() const;

        std::vector<int> CopyIndptr() const;
        std::vector<int> CopyIndices() const;
        std::vector<T> CopyData() const;

        void Print(const std::string& label = "") const;
        void PrintDense(const std::string& label = "") const;

        DenseMatrix ToDense() const;

        void SortIndices();

        template <typename T2 = T>
        auto Mult(const Vector<T2>& input) const;

        template <typename T2 = T>
        auto MultAT(const Vector<T2>& input) const;

        template <typename T2 = T, typename T3 = T>
        void Mult(const Vector<T2>& input, Vector<T3>& output) const;

        template <typename T2 = T, typename T3 = T>
        void MultAT(const Vector<T2>& input, Vector<T3>& output) const;

        DenseMatrix Mult(const DenseMatrix& input) const;
        DenseMatrix MultAT(const DenseMatrix& input) const;

        void Mult(const DenseMatrix& input, DenseMatrix& output) const;
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;

        template <typename T2 = T>
        auto Mult(const SparseMatrix<T2>& rhs) const;

        SparseMatrix<T> Transpose() const;

        std::vector<double> GetDiag() const;

        SparseMatrix<T> GetSubMatrix(const std::vector<int>& rows,
                                     const std::vector<int>& cols,
                                     std::vector<int>& marker) const;
    private:
        int rows_;
        int cols_;
        int nnz_;

        std::vector<int> indptr_;
        std::vector<int> indices_;
        std::vector<T> data_;
};

template <typename T>
SparseMatrix<T>::SparseMatrix()
    : rows_(0), cols_(0), nnz_(0), indptr_(0), indices_(0), data_(0)
{

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

    assert(static_cast<int>(indptr_.size()) == rows_ + 1);
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
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T>& other)
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
      indptr_(other.indptr_), indices_(other.indices_), data_(other.data_)
{

}


template <typename T>
SparseMatrix<T>::SparseMatrix(SparseMatrix<T>&& other)
{
    Swap(*this, other);
}

template <typename T>
SparseMatrix<T>& SparseMatrix<T>::operator=(SparseMatrix<T> other)
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
void SparseMatrix<T>::Print(const std::string& label) const
{
    constexpr int width = 6;

    std::cout << label << "\n";

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            std::cout << std::setw(width) << "(" <<
                      i << ", " << indices_[j] << ") " << data_[j] << "\n";
        }
    }

    std::cout << "\n";
}

template <typename T>
void SparseMatrix<T>::PrintDense(const std::string& label) const
{
    const DenseMatrix dense = ToDense();

    dense.Print(label);
}

template <typename T>
DenseMatrix SparseMatrix<T>::ToDense() const
{
    DenseMatrix dense(rows_, cols_);

    for (int i = 0; i < rows_; ++i)
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

    for (int i = 0; i < rows_; ++i)
    {
        const int start = indptr_[i];
        const int end = indptr_[i + 1];

        std::sort(begin(permutation) + start,
                  begin(permutation) + end,
                  compare_cols);
    }

    std::swap(indices_, permutation);
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

    for (int k = 0; k < input.Cols(); ++k)
    {
        for (int i = 0; i < rows_; ++i)
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

    for (int k = 0; k < input.Cols(); ++k)
    {
        for (int i = 0; i < rows_; ++i)
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

    for (int i = 0; i < cols_; ++i)
    {
        out_indptr[i + 1] += out_indptr[i];
    }

    for (int i = 0; i < rows_; ++i)
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

    for (int i = 0; i < rows_; ++i)
    {
        double val = 0.0;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (indices_[j] == i)
            {
                val = data_[j];
            }
        }

        diag[i] = val;
    }

    return diag;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::GetSubMatrix(const std::vector<int>& rows,
                                              const std::vector<int>& cols,
                                              std::vector<int>& marker) const
{
    assert(marker.size() >= static_cast<unsigned int>(cols_));

    std::vector<int> out_indptr(rows.size() + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    const int out_rows = rows.size();
    const int out_cols = cols.size();

    for (int i = 0; i < out_cols; ++i)
    {
        marker[cols[i]] = i;
    }

    for (int i = 0; i < out_rows; ++i)
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
int SparseMatrix<T>::Rows() const
{
    return rows_;
}

template <typename T>
inline
int SparseMatrix<T>::Cols() const
{
    return cols_;
}

template <typename T>
inline
int SparseMatrix<T>::nnz() const
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
std::vector<int> SparseMatrix<T>::CopyIndptr() const
{
    return indptr_;
}

template <typename T>
inline
std::vector<int> SparseMatrix<T>::CopyIndices() const
{
    return indices_;
}

template <typename T>
inline
std::vector<T> SparseMatrix<T>::CopyData() const
{
    return data_;
}


template <typename T>
template <typename T2>
auto SparseMatrix<T>::Mult(const Vector<T2>& input) const
{
    Vector<typename std::common_type<T, T2>::type> output(rows_);
    Mult(input, output);

    return output;
}

template <typename T>
template <typename T2>
auto SparseMatrix<T>::MultAT(const Vector<T2>& input) const
{
    Vector<typename std::common_type<T, T2>::type> output(cols_);
    MultAT(input, output);

    return output;
}

template <typename T>
template <typename T2, typename T3>
void SparseMatrix<T>::Mult(const Vector<T2>& input, Vector<T3>& output) const
{
    assert(input.size() == cols_);
    assert(output.size() == rows_);

    for (int i = 0; i < rows_; ++i)
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
void SparseMatrix<T>::MultAT(const Vector<T2>& input, Vector<T3>& output) const
{
    assert(input.size() == rows_);
    assert(output.size() == cols_);

    std::fill(std::begin(output), std::end(output), 0.0);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            output[indices_[j]] += data_[j] * input[i];
        }
    }
}

template <typename T>
template <typename T2>
auto SparseMatrix<T>::Mult(const SparseMatrix<T2>& rhs) const
{
    std::vector<int> marker(rhs.cols_, -1);

    std::vector<int> out_indptr(rows_ + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs.indptr_[indices_[j]]; k < rhs.indptr_[indices_[j] + 1]; ++k)
            {
                if (marker[rhs.indices_[k]] != i)
                {
                    marker[rhs.indices_[k]] = i;
                    ++out_nnz;
                }
            }
        }

        out_indptr[i + 1] = out_nnz;
    }

    std::fill(begin(marker), end(marker), -1);

    std::vector<int> out_indices(out_nnz);
    std::vector<typename std::common_type<T, T2>::type> out_data(out_nnz);

    int total = 0;

    for (int i = 0; i < rows_; ++i)
    {
        int row_nnz = total;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs.indptr_[indices_[j]]; k < rhs.indptr_[indices_[j] + 1]; ++k)
            {
                if (marker[rhs.indices_[k]] < row_nnz)
                {
                    marker[rhs.indices_[k]] = total;
                    out_indices[total] = rhs.indices_[k];
                    out_data[total] = data_[j] * rhs.data_[k];

                    total++;
                }
                else
                {
                    out_data[marker[rhs.indices_[k]]] += data_[j] * rhs.data_[k];
                }
            }
        }
    }

    return SparseMatrix<typename std::common_type<T, T2>::type>(out_indptr, out_indices, out_data,
                                                                rows_, rhs.cols_);
}

} //namespace linalgcpp

#endif // SPARSEMATRIX_HPP__
