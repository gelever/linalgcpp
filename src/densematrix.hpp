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
		DenseMatrix(DenseMatrix&&) = default;
		~DenseMatrix() = default;

		void Print(const std::string& lable = "") const;

		double& operator()(int row, int col);
		const double& operator()(int row, int col) const;

		void Mult(const std::vector<double>& input, std::vector<double>& output);

	private:
		int rows_;
		int cols_;
		std::vector<double> data_;

};

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
