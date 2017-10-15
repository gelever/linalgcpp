#include "sparsematrix.hpp"

SparseMatrix::SparseMatrix()
	: rows_(0), cols_(0), nnz_(0)
{

}

SparseMatrix::SparseMatrix(const std::vector<int>& indptr,
		           const std::vector<int>& indices,
			   const std::vector<double>& data,
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

std::vector<double> SparseMatrix::Mult(const std::vector<double>& input) const
{
	std::vector<double> output(rows_);
	Mult(input, output);

	return output;
}

std::vector<double> SparseMatrix::MultAT(const std::vector<double>& input) const
{
	std::vector<double> output(cols_);
	MultAT(input, output);

	return output;
}

void SparseMatrix::Mult(const std::vector<double>& input, std::vector<double>& output) const
{
	assert(input.size() == static_cast<unsigned int>(cols_));
	assert(output.size() == static_cast<unsigned int>(rows_));

	for (int i = 0; i < rows_; ++i)
	{
		double val = 0.0;

		for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
		{
			val += data_[j] * input[indices_[j]];
		}

		output[i] = val;
	}
}

void SparseMatrix::MultAT(const std::vector<double>& input, std::vector<double>& output) const
{
	assert(input.size() == static_cast<unsigned int>(rows_));
	assert(output.size() == static_cast<unsigned int>(cols_));

	std::fill(begin(output), end(output), 0.0);

	for (int i = 0; i < rows_; ++i)
	{
		for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
		{
			output[indices_[j]] += data_[j] * input[i];
		}
	}
}

DenseMatrix SparseMatrix::Mult(const DenseMatrix& input) const
{
	DenseMatrix output(rows_, input.Cols());
	Mult(input, output);

	return output;
}

DenseMatrix SparseMatrix::MultAT(const DenseMatrix& input) const
{
	DenseMatrix output(cols_, input.Cols());
	MultAT(input, output);

	return output;
}

void SparseMatrix::Mult(const DenseMatrix& input, DenseMatrix& output) const
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

void SparseMatrix::MultAT(const DenseMatrix& input, DenseMatrix& output) const
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

SparseMatrix SparseMatrix::Mult(const SparseMatrix& rhs) const
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
	std::vector<double> out_data(out_nnz);

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

	return SparseMatrix(out_indptr, out_indices, out_data,
			    rows_, rhs.cols_);
}

SparseMatrix SparseMatrix::Transpose() const
{
	std::vector<int> out_indptr(cols_ + 1, 0);
	std::vector<int> out_indices(nnz_);
	std::vector<double> out_data(nnz_);

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
			const int val = data_[j];

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

std::vector<double> SparseMatrix::GetDiag() const
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

SparseMatrix SparseMatrix::GetSubMatrix(const std::vector<int>& rows,
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
	std::vector<double> out_data(out_nnz);

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

	return SparseMatrix(out_indptr, out_indices, out_data,
			    out_rows, out_cols);
}







