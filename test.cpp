#include <random>
#include <stdio.h>
#include <assert.h>

#include "src/vector.hpp"
#include "src/densematrix.hpp"
#include "src/sparsematrix.hpp"
#include "src/coomatrix.hpp"

void test_sparse()
{
	const int size = 3;
	const int nnz = 5;

	std::vector<int> indptr(size + 1);
	std::vector<int> indices(nnz);
	std::vector<double> data(nnz);

	indptr[0] = 0;
	indptr[1] = 2;
	indptr[2] = 3;
	indptr[3] = 5;

	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 0;
	indices[3] = 1;
	indices[4] = 2;

	data[0] = 1;
	data[1] = 2;
	data[2] = 3;
	data[3] = 4;
	data[4] = 5;

	SparseMatrix A(indptr, indices, data, size, size);
	A.PrintDense("A:");

	std::vector<double> x(size, 1.0);

	auto y = A.Mult(x);
	auto yt = A.MultAT(x);

	printf("x:");
	std::cout << x;
	printf("y:");
	std::cout << y;
	printf("yt:");
	std::cout << yt;

	DenseMatrix rhs(size);

	rhs(0, 0) = 1.0;
	rhs(1, 1) = 2.0;
	rhs(2, 2) = 3.0;

	rhs.Print("rhs");

	auto ab = A.Mult(rhs);
	ab.Print("ab:");

	auto ba = A.MultAT(rhs);
	ba.Print("ba:");

	auto B = A;

	auto C = A.Mult(B);
	C.PrintDense("C:");

	auto C2 = A.ToDense().Mult(B.ToDense());
	C2.Print("C dense:");

	auto AT = A.Transpose();
	AT.PrintDense("AT:");

	std::vector<int> rows({0, 2});
	std::vector<int> cols({0, 2});
	std::vector<int> marker(size, -1);


	auto submat = A.GetSubMatrix(rows, cols, marker);

	A.PrintDense("A:");
	submat.PrintDense("Submat");

	{
		const int size = 1e5;
		const int sub_size = 1e2;
		const int num_entries = 5e6;

		CooMatrix coo(size);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, size - 1);

		std::vector<int> rows(sub_size);
		std::vector<int> cols(sub_size);
		std::vector<int> marker(size, -1);

		for (int iter = 0; iter < num_entries; ++iter)
		{
			int i = dis(gen);
			int j = dis(gen);
			double val = dis(gen);

			coo.Add(i, j, val);
		}

		auto sparse = coo.ToSparse();

		for (int i = 0; i < sub_size; ++i)
		{
			rows[i] = dis(gen);
			cols[i] = dis(gen);
		}

		auto submat = sparse.GetSubMatrix(rows, cols, marker);
		printf("%d %d %d\n", submat.Rows(), submat.Cols(), submat.nnz());

		//submat.PrintDense("submat:");
		submat.Print("submat:");

	}



}
void test_coo()
{
	// Without setting specfic size
	{
		CooMatrix coo(10, 10);
		coo.Add(0, 0, 1.0);
		coo.Add(0, 1, 2.0);
		coo.Add(1, 1, 3.0);
		coo.Add(1, 1, 3.0);
		coo.Add(1, 1, 3.0);
		coo.Add(2, 2, 3.0);
		coo.Add(4, 2, 3.0);

		auto dense = coo.ToDense();
		auto sparse = coo.ToSparse();

	}
	// Without setting specfic size
	{
		CooMatrix coo;
		coo.Add(0, 0, 1.0);
		coo.Add(0, 1, 2.0);
		coo.Add(1, 1, 3.0);
		coo.Add(1, 1, 3.0);
		coo.Add(1, 1, 3.0);
		coo.Add(2, 2, 3.0);
		coo.Add(4, 2, 3.0);

		auto dense = coo.ToDense();
		auto sparse = coo.ToSparse();
		auto diff = dense - sparse.ToDense();

		assert(std::fabs(diff.Sum()) < 1e-8);
	}
	{
		CooMatrix coo(10, 10);

		std::vector<int> rows({8, 0, 3});
		std::vector<int> cols({6, 4, 8});

		DenseMatrix input(3, 3);
		input(0, 0) = 1.0;
		input(0, 1) = 2.0;
		input(0, 2) = 3.0;
		input(1, 0) = 4.0;
		input(1, 1) = 5.0;
		input(1, 2) = 6.0;
		input(2, 0) = 7.0;
		input(2, 1) = 8.0;
		input(2, 2) = 9.0;

		coo.Add(rows, cols, input);

		auto sparse = coo.ToSparse();
		auto dense = coo.ToDense();
		auto diff = dense - sparse.ToDense();

		assert(std::fabs(diff.Sum()) < 1e-8);
	}
	{
		const int size = 1e1;
		const int num_entries = 1e2;

		CooMatrix coo(size);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, size - 1);

		for (int iter = 0; iter < num_entries; ++iter)
		{
			int i = dis(gen);
			int j = dis(gen);
			double val = dis(gen);

			coo.Add(i, j, val);
		}

		auto sparse = coo.ToSparse();
		auto dense = coo.ToDense();
		auto diff = dense - sparse.ToDense();

		assert(std::fabs(diff.Sum()) < 1e-8);
	}
}

void test_dense()
{
	const int size = 5;

	DenseMatrix d1;
	DenseMatrix d2(size);
	DenseMatrix d3(size, size);
	DenseMatrix d4(d3);

	d2(0, 0) = 0.0;
	d2(1, 1) = 1.0;
	d2(0, 1) = 1.0;
	d2(1, 0) = 1.0;
	d2(2, 2) = 2.0;
	d2(2, 0) = 2.0;
	d2(0, 2) = 2.0;
	d2(3, 3) = 3.0;
	d2(0, 3) = 3.0;
	d2(3, 0) = 3.0;
	d2(4, 4) = 4.0;
	d2(4, 0) = 4.0;
	d2(0, 4) = 4.0;

	d2.Print();

	std::vector<double> x(size, 1.0);
	std::vector<double> y(size);

	d2.Mult(x, y);

	printf("d2 * x = y:\n");
	std::cout << y;

	printf("d2 * y:\n");
	d2.MultAT(y, x);

	std::cout << x;

	DenseMatrix A(3, 2);
	DenseMatrix B(2, 4);

	A(0, 0) = 1.0;
	A(1, 1) = 2.0;
	A(2, 0) = 3.0;

	B(0, 0) = 1.0;
	B(0, 2) = 2.0;
	B(1, 1) = 3.0;
	B(1, 3) = 4.0;

	A.Print("A:");
	B.Print("B:");

	DenseMatrix C = A.Mult(B);

	C.Print("C:");

	DenseMatrix D = A.MultAT(C);
	D.Print("D:");

	DenseMatrix E = C.MultBT(B);
	E.Print("E:");

	DenseMatrix F = B.MultABT(A);
	F.Print("F:");

	F *= 2.0;
	F.Print("2F:");
	F /= 2.0;
	F.Print("F:");

	DenseMatrix G = 5 * F;
	DenseMatrix G2 = F * 5;
	G.Print("5 *F:");
	G2.Print("F *5:");




}

void test_vector()
{
	const int size = 5;

	std::vector<double> d1(size, 1.0);
	std::vector<double> d2(size, 2.0);

	std::vector<double> d3 = d1 + d2;
	std::vector<double> d4 = d1 - d2;
	std::vector<double> d5 = 5.0 * d1;
	std::vector<double> d6 = d1;
	std::vector<double> d7 = d2;
	d6 *= 1.5;
	d7 /= 1.5;


	std::cout << d1;
	std::cout << d2;
	std::cout << d3;
	std::cout << d4;
	std::cout << d5;
	std::cout << d6;
	std::cout << d7;
	std::cout << d4 * d3 << "\n";
}

int main(int argc, char** argv)
{
	test_vector();
	test_dense();
	test_coo();
	test_sparse();

	return EXIT_SUCCESS;
}
