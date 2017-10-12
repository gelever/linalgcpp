#include "densematrix.hpp"

DenseMatrix::DenseMatrix()
	: rows_(0), cols_(0)
{
}

DenseMatrix::DenseMatrix(int size)
	: DenseMatrix(size, size)
{
}

DenseMatrix::DenseMatrix(int rows, int cols)
	: rows_(rows), cols_(cols), data_(rows * cols, 0.0)
{
	assert(rows >= 0);
	assert(cols >= 0);
}

DenseMatrix::DenseMatrix(int rows, int cols, const std::vector<double>& data)
	: rows_(rows), cols_(cols), data_(data)
{
	assert(rows >= 0);
	assert(cols >= 0);

	assert(data.size() == rows * cols);
}

void DenseMatrix::Print(const std::string& label) const
{
	std::cout << label << "\n";

	const int width = 6;

	for (int i = 0; i < rows_; ++i)
	{
		for (int j = 0; j < cols_; ++j)
		{
			std::cout << std::setw(width) << ((*this)(i, j));

		}

		std::cout << "\n";
	}
	
	std::cout << "\n";
}

void DenseMatrix::Mult(const std::vector<double>& input, std::vector<double>& output)
{
	assert(input.size() == static_cast<unsigned int>(cols_));
	assert(output.size() == static_cast<unsigned int>(rows_));
	std::fill(begin(output), end(output), 0.0);

	for (int j = 0; j < cols_; ++j)
	{
		for (int i = 0; i < rows_; ++i)
		{
			output[i] += (*this)(i, j) * input[j];
		}
	}
}



