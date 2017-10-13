#include "sparsematrix.hpp"

SparseMatrix::SparseMatrix()
	: rows_(0), cols_(0)
{

}

SparseMatrix::SparseMatrix(const std::vector<int>& indptr,
		           const std::vector<int>& indices,
			   const std::vector<double>& data,
			   int rows, int cols)
	: rows_(rows), cols_(cols),
	indptr_(indptr), indices_(indices), data_(data)
{
	assert(rows_ >= 0);
	assert(cols_ >= 0);

	assert(static_cast<int>(indptr_.size()) == rows_ + 1);
	assert(indices_.size() == data_.size());
	assert(indptr_[0] == 0);
}

SparseMatrix::SparseMatrix(const std::vector<double>& diag)
	: rows_(diag.size()), cols_(diag.size()), nnz_(diag.size()),
	indptr_(diag.size() + 1), indices_(diag.size()), data_(diag)
{
	std::iota(begin(indptr_), end(indptr_), 0);
	std::iota(begin(indices_), end(indices_), 0);
}


SparseMatrix::SparseMatrix(SparseMatrix&& other)
{
	std::swap(*this, other);
}

SparseMatrix& SparseMatrix::operator=(SparseMatrix other)
{
	std::swap(*this, other);

	return *this;
}

void SparseMatrix::Print(const std::string& label) const
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

void SparseMatrix::PrintDense(const std::string& label) const
{
	const DenseMatrix dense = ToDense();

	dense.Print(label);
}

DenseMatrix SparseMatrix::ToDense() const
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

void SparseMatrix::SortIndices()
{
	const auto compare_cols = [&](int i, int j) {
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
