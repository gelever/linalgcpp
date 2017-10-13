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

class CooMatrix
{
	public:
		CooMatrix();
		CooMatrix(int size);
		CooMatrix(int rows, int cols);

		void Add(int i, int j, double val);

		SparseMatrix ToSparse() const;
		DenseMatrix ToDense() const;

	private:
		int rows_;
		int cols_;

		std::priority_queue<std::tuple<int, int, double>> queue;
		std::vector<std::tuple<int, int, double>> vqueue;
};

inline
CooMatrix::CooMatrix()
	: rows_(-1), cols_(-1)
{
}

inline
CooMatrix::CooMatrix(int size) : CooMatrix(size, size)
{

}

inline
CooMatrix::CooMatrix(int rows, int cols)
	: rows_(rows), cols_(cols)
{

}

inline
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

	vqueue.push_back(std::make_tuple(i, j, val));


	constexpr auto cmp = [](auto lhs, auto rhs)
	{
		return lhs > rhs;
	};

	std::push_heap(begin(vqueue), end(vqueue), cmp);
}

inline
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
		double val;
		std::tie(rows, cols, val) = vqueue.back();
	}

	// Increase to make sizes 1-based
	rows++;
	cols++;

	DenseMatrix dense(rows, cols);

	for (const auto& tup : vqueue)
	{
		int i;
		int j;
		double val;

		std::tie(i, j, val) = tup;

		dense(i, j) += val;
	}

	return dense;
}

inline
SparseMatrix CooMatrix::ToSparse() const
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
		double val;
		std::tie(rows, cols, val) = vqueue.back();
	}

	// Increase to make sizes 1-based
	rows++;
	cols++;

	std::vector<int> indptr(rows + 1);
	std::vector<int> indices;
	std::vector<double> data;

	int current_row = 0;

	for (const auto& tup : vqueue)
	{
		int i;
		int j;
		double val;

		std::tie(i, j, val) = tup;

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

	std::fill(begin(indptr) + current_row,
			end(indptr), data.size());

	return SparseMatrix(indptr, indices, data, rows, cols);
}


#endif // COOMATRIX_HPP__
