#include <random>
#include <stdio.h>
#include <assert.h>

#include "src/linalgcpp.hpp"

using namespace linalgcpp;

void test_sparse()
{
    const int size = 3;
    const int nnz = 5;

    SparseMatrix<double> A;

    {
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

        A = SparseMatrix<double>(indptr, indices, data, size, size);

        SparseMatrix<double> test(indptr, indices, data, size, size);
        SparseMatrix<double> test2(std::move(test));
    }

    A.PrintDense("A:");

    SparseMatrix<int> A_int;
    {
        std::vector<int> indptr(size + 1);
        std::vector<int> indices(nnz);
        std::vector<int> data(nnz);

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

        A_int = SparseMatrix<int>(indptr, indices, data, size, size);
    }

    A_int.PrintDense("A_int:");

    auto AA = A.Mult(A);
    AA.PrintDense("A*A:");

    SparseMatrix<> AA_int = A.Mult(A);

    Vector<double> x(size, 1.0);
    Vector<double> y = A.Mult(x);
    Vector<double> yt = A.MultAT(x);

    printf("x:");
    std::cout << x;
    printf("Ax = y:");
    std::cout << y;
    printf("A^T x = y:");
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
        const int size = 1e2;
        const int sub_size = 1e1;
        const int num_entries = 5e3;

        CooMatrix<double> coo(size);

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

        CooMatrix<double> coo2 = coo;
        auto sparse2 = coo2.ToSparse();

        //submat.PrintDense("submat:");
        //submat.Print("submat:");

    }



}
void test_coo()
{
    // Without setting specfic size
    {
        CooMatrix<double> coo(10, 10);
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
        CooMatrix<double> coo;
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
        CooMatrix<double> coo(10, 10);

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

        CooMatrix<double> coo(size);

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
    //std::cout << y;

    printf("d2 * y:\n");
    d2.MultAT(y, x);

    //std::cout << x;

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

    Vector<double> v1;
    Vector<double> v2(size);
    Vector<double> v3(size, 3.0);

    std::cout << "v1:";
    std::cout << v1;
    std::cout << "v2:";
    std::cout << v2;
    std::cout << "v3:";
    std::cout << v3;

    Normalize(v3);
    std::cout << "v3 normalized:";
    std::cout << v3;

    std::cout << "v3[0]:" << v3[0] << "\n";

    auto v4 = v3 * v3;
    std::cout << "v3 * v3: " << v4 << "\n";
}

int main(int argc, char** argv)
{
    test_sparse();
    test_dense();
    test_coo();
    test_vector();

    return EXIT_SUCCESS;
}
