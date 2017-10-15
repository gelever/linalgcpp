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
		void Add(const std::vector<int>& rows,
				const std::vector<int>& cols,
				const DenseMatrix& values);

		SparseMatrix ToSparse() const;
		DenseMatrix ToDense() const;

	private:
		int rows_;
		int cols_;

		mutable std::vector<std::tuple<int, int, double>> entries;

};

#endif // COOMATRIX_HPP__
