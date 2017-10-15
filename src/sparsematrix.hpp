#ifndef SPARSEMATRIX_HPP__
#define SPARSEMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "densematrix.hpp"

class SparseMatrix
{
	public:
		SparseMatrix();
		SparseMatrix(const std::vector<int>& indptr,
		             const std::vector<int>& indices,
			     const std::vector<double>& data,
			     int rows, int cols);

		SparseMatrix(const std::vector<double>& diag);

		SparseMatrix(const SparseMatrix& other) = default;
		~SparseMatrix() noexcept = default;

		SparseMatrix& operator=(SparseMatrix other);
		SparseMatrix(SparseMatrix&& other);
		friend void Swap(SparseMatrix& lhs, SparseMatrix& rhs);

		int Rows() const;
		int Cols() const;
		int nnz() const;

		const std::vector<int>& GetIndptr() const;
		const std::vector<int>& GetIndices() const;
		const std::vector<double>& GetData() const;

		std::vector<int> CopyIndptr() const;
		std::vector<int> CopyIndices() const;
		std::vector<double> CopyData() const;

		void Print(const std::string& label = "") const;
		void PrintDense(const std::string& label = "") const;

		DenseMatrix ToDense() const;

		void SortIndices();

		std::vector<double> Mult(const std::vector<double>& input) const;
		std::vector<double> MultAT(const std::vector<double>& input) const;

		void Mult(const std::vector<double>& input, std::vector<double>& output) const;
		void MultAT(const std::vector<double>& input, std::vector<double>& output) const;

		DenseMatrix Mult(const DenseMatrix& input) const;
		DenseMatrix MultAT(const DenseMatrix& input) const;

		void Mult(const DenseMatrix& input, DenseMatrix& output) const;
		void MultAT(const DenseMatrix& input, DenseMatrix& output) const;

		SparseMatrix Mult(const SparseMatrix& rhs) const;

		SparseMatrix Transpose() const;

		std::vector<double> GetDiag() const;

		SparseMatrix GetSubMatrix(const std::vector<int>& rows,
				          const std::vector<int>& cols,
					  std::vector<int>& marker) const;



	private:
		int rows_;
		int cols_;
		int nnz_;

		std::vector<int> indptr_;
		std::vector<int> indices_;
		std::vector<double> data_;
};

inline
int SparseMatrix::Rows() const
{
	return rows_;
}

inline
int SparseMatrix::Cols() const
{
	return cols_;
}

inline
int SparseMatrix::nnz() const
{
	return nnz_;
}

inline
const std::vector<int>& SparseMatrix::GetIndptr() const
{
	return indptr_;
}

inline
const std::vector<int>& SparseMatrix::GetIndices() const
{
	return indices_;
}

inline
const std::vector<double>& SparseMatrix::GetData() const
{
	return data_;
}

inline
std::vector<int> SparseMatrix::CopyIndptr() const
{
	return indptr_;
}

inline
std::vector<int> SparseMatrix::CopyIndices() const
{
	return indices_;
}

inline
std::vector<double> SparseMatrix::CopyData() const
{
	return data_;
}


#endif // SPARSEMATRIX_HPP__
