#ifndef DENSEMATRIX_HPP__
#define DENSEMATRIX_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "vector.hpp"

class DenseMatrix
{
	public:
		DenseMatrix();
		DenseMatrix(int size);
		DenseMatrix(int rows, int cols);
		DenseMatrix(int rows, int cols, const std::vector<double>& data);

		DenseMatrix(const DenseMatrix&) = default;

		friend void Swap(DenseMatrix& lhs, DenseMatrix& rhs);
		DenseMatrix(DenseMatrix&&);
		~DenseMatrix() = default;

		void Print(const std::string& label = "") const;

		double& operator()(int row, int col);
		const double& operator()(int row, int col) const;

		void Mult(const std::vector<double>& input, std::vector<double>& output);

		int Rows() const;
		int Cols() const;

	private:
		int rows_;
		int cols_;
		std::vector<double> data_;

};

inline
int DenseMatrix::Rows() const
{
	return rows_;
}

inline
int DenseMatrix::Cols() const
{
	return cols_;
}

inline
double& DenseMatrix::operator()(int row, int col)
{
	assert(row >= 0);
	assert(col >= 0);

	assert(row < rows_);
	assert(col < cols_);

	return data_[row + (col * rows_)];
}

inline
const double& DenseMatrix::operator()(int row, int col) const
{
	assert(row >= 0);
	assert(col >= 0);

	assert(row < rows_);
	assert(col < cols_);

	return data_[row + (col * rows_)];
}

#endif // DENSEMATRIX_HPP__
