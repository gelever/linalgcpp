#include "coomatrix.hpp"

CooMatrix::CooMatrix()
	: rows_(-1), cols_(-1)
{
}

CooMatrix::CooMatrix(int size) : CooMatrix(size, size)
{

}

CooMatrix::CooMatrix(int rows, int cols)
	: rows_(rows), cols_(cols)
{

}

void CooMatrix::Add(int i, int j, double val)
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

void CooMatrix::Add(const std::vector<int>& rows,
		    const std::vector<int>& cols,
		    const DenseMatrix& values)
{
	assert(rows.size() == values.Rows());
	assert(cols.size() == values.Cols());

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

DenseMatrix CooMatrix::ToDense() const
{
	int rows;
	int cols;

	if (rows_ > -1)
	{
		rows = rows_;
		cols = cols_;
	}
	else
	{
		auto max_el = Max(entries);
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

SparseMatrix CooMatrix::ToSparse() const
{
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
	std::vector<double> data;

	indptr[0] = 0;

	int current_row = 0;

	for (const auto& tup : entries)
	{
		const int i = std::get<0>(tup);
		const int j = std::get<1>(tup);
		const double val = std::get<2>(tup);

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

	return SparseMatrix(indptr, indices, data, rows, cols);
}
