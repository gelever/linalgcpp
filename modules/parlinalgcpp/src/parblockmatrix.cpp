/*! @file */

#include "parblockmatrix.hpp"

namespace linalgcpp
{

ParBlockMatrix::ParBlockMatrix()
    : row_offsets_(1, 0), col_offsets_(1, 0)
{

}

ParBlockMatrix::ParBlockMatrix(MPI_Comm comm, std::vector<int> offsets)
  : ParBlockMatrix(comm, offsets, offsets)
{

}

ParBlockMatrix::ParBlockMatrix(MPI_Comm comm, std::vector<int> row_offsets, std::vector<int> col_offsets)
    : ParOperator(comm),
      row_offsets_(std::move(row_offsets)), col_offsets_(std::move(col_offsets)),
      A_(row_offsets_.size() - 1, std::vector<ParMatrix>(col_offsets_.size() - 1))
{
    auto starts = GenerateOffsets(comm, {row_offsets_.back(), col_offsets_.back()});
    assert(starts.size() == 2);

    std::swap(row_starts_, starts[0]);
    std::swap(col_starts_, starts[1]);

    InitVector(x_, row_starts_);
    InitVector(b_, col_starts_);
}

ParBlockMatrix::ParBlockMatrix(const ParBlockMatrix& other) noexcept
    : ParOperator(other), row_offsets_(other.row_offsets_), col_offsets_(other.col_offsets_),
      A_(other.A_)
{

}

ParBlockMatrix::ParBlockMatrix(ParBlockMatrix&& other) noexcept
{
    swap(*this, other);
}

ParBlockMatrix& ParBlockMatrix::operator=(ParBlockMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(ParBlockMatrix& lhs, ParBlockMatrix& rhs) noexcept
{
    swap(static_cast<ParOperator&>(lhs), static_cast<ParOperator&>(rhs));

    std::swap(lhs.row_offsets_, rhs.row_offsets_);
    std::swap(lhs.col_offsets_, rhs.col_offsets_);
    swap(lhs.A_, rhs.A_);
}

const std::vector<int>& ParBlockMatrix::GetRowOffsets() const
{
    return row_offsets_;
}

const std::vector<int>& ParBlockMatrix::GetColOffsets() const
{
    return col_offsets_;
}

const ParMatrix& ParBlockMatrix::GetBlock(int i, int j) const
{
    return A_.at(i).at(j);
}

void ParBlockMatrix::SetBlock(int i, int j, ParMatrix mat)
{
    assert(i < static_cast<int>(row_offsets_.size()) - 1);
    assert(j < static_cast<int>(col_offsets_.size()) - 1);

    assert(mat.Rows() == (row_offsets_[i + 1] - row_offsets_[i]));
    assert(mat.Cols() == (col_offsets_[j + 1] - col_offsets_[j]));

    swap(A_[i][j], mat);
}

void ParBlockMatrix::Mult(const VectorView<double>& input,
                          VectorView<double> output) const
{

}

void ParBlockMatrix::MultAT(const VectorView<double>& input,
                            VectorView<double> output) const
{

}

ParBlockMatrix ParBlockMatrix::Transpose() const
{
    ParBlockMatrix transpose(GetComm(), col_offsets_, row_offsets_);

    const int row_blocks = row_offsets_.size() - 1;
    const int col_blocks = col_offsets_.size() - 1;

    for (int i = 0; i < row_blocks; ++i)
    {
        for (int j = 0; j < col_blocks; ++j)
        {
            const ParMatrix& A_ij = GetBlock(i, j);

            if (A_ij.Rows() > 0 && A_ij.Cols() > 0)
            {
                transpose.SetBlock(j, i, A_ij.Transpose());
            }
        }
    }

    return transpose;
}

} // namespace linalgcpp
