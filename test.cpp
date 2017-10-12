#include <stdio.h>
#include "src/vector.hpp"
#include "src/densematrix.hpp"

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

	std::cout << y;




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

	return EXIT_SUCCESS;
}
