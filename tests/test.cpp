/*! @file

    @brief A collection of brief tests to make sure
          things do what I expect them.  None of these
          checks are automated yet, but will be in the near
          future.
*/
#include <random>
#include <stdio.h>
#include <assert.h>

#include "linalgcpp.hpp"

using namespace linalgcpp;

void test_lil()
{
    LilMatrix<double> A;

    {
        LilMatrix<int> lil(3, 3);
        lil.AddSym(0, 0, 1);
        lil.AddSym(0, 1, 2);
        lil.AddSym(0, 2, 3);

        LilMatrix<int> lil2(lil);
        lil2.AddSym(0, 0, 1);

        LilMatrix<int> lil3;
        lil3 = lil2;
        lil3.AddSym(0, 0, 1);

        constexpr int size = 100;
        LilMatrix<int> lil_large(size);

        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; j += 2)
            {
                lil_large.AddSym(i, j, i * j);
                lil_large.AddSym(j, i, i * j);
            }
        }

        auto lil_sparse = lil.ToSparse();
        auto lil_sparse2 = lil2.ToSparse();
        auto lil_sparse3 = lil3.ToSparse();
        auto lil_sparse_large = lil_large.ToSparse();

        lil_sparse.Print("Lil:");
        lil_sparse2.Print("Lil2:");
        lil_sparse3.Print("Lil3:");

        auto lil_dense = lil.ToDense();
        auto lil_dense2 = lil2.ToDense();
        auto lil_dense3 = lil3.ToDense();

        lil_dense.Print("Lil:");
        lil_dense2.Print("Lil2:");
        lil_dense3.Print("Lil3:");

        LilMatrix<double> lil_zero(size);
        lil_zero.Add(2, 2, 1e-8);
        lil_zero.Add(2, 1, 1e-5);
        lil_zero.Print("Lil3: with zero");
        lil_zero.EliminateZeros(1e-6);
        lil_zero.Print("Lil3: No zero");
    }

}
void test_sparse()
{
    // test empty
    {
        SparseMatrix<double> A;
        SparseMatrix<double> A2(10);
        SparseMatrix<double> A3(10, 10);


        Vector<double> x(10, 1.0);
        auto y = A2.Mult(x);

        assert(AbsMin(y) == 0.0);
        assert(AbsMax(y) == 0.0);
    }

    // Test create diag
    {
        std::vector<double> data(3, 3.0);
        SparseMatrix<double> A(data);
        std::cout << "0 1 2 3: "  <<  A.GetIndptr();
        std::cout << "0 1 2 : " << A.GetIndices();
        std::cout << "3 3 3 : " << A.GetData();
    }

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

        data[0] = 1.5;
        data[1] = 2.5;
        data[2] = 3.5;
        data[3] = 4.5;
        data[4] = 5.5;

        A = SparseMatrix<double>(indptr, indices, data, size, size);

        SparseMatrix<double> test(indptr, indices, data, size, size);
        SparseMatrix<double> test2(std::move(test));
    }

    A.Print("A:");
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

    auto AA_double = A.Mult(A);
    auto AA_auto = A.Mult(A_int);
    auto AA_int = A_int.Mult(A_int);
    auto AA_force_int = A.Mult<double, int>(A);
    auto AA_force_double = A_int.Mult<int, double>(A_int);

    AA_double.PrintDense("A_double *A_double  double:");
    AA_auto.PrintDense("A_double *A_int  double:");
    AA_int.PrintDense("A_int *A_int  int:");
    AA_force_int.PrintDense("A_double *A_double forced to int:");
    AA_force_double.PrintDense("A_int *A_int forced to double:");
    AA_force_double *= 1.1;
    AA_force_double.PrintDense("A_int *A_int forced to double * 1.1:");

    Vector<double> x(size, 1.5);
    Vector<double> y = A.Mult(x);
    Vector<double> yt = A.MultAT(x);

    // printf("x:");
    // std::cout << x;
    // printf("Ax = y:");
    // std::cout << y;
    // printf("A^T x = y:");
    // std::cout << yt;

    Vector<int> x_int(size, 1.0);
    auto y_auto = A.Mult(x_int);
    auto y_auto_int = A_int.Mult(x_int);
    auto y_auto_dub = A_int.Mult(x);

    y_auto.Print("y_auto");
    y_auto_int.Print("y_auto_int");
    y_auto_dub.Print("y_auto_dub");

    DenseMatrix rhs(size, 2);

    rhs(0, 0) = 1.0;
    rhs(0, 1) = 2.0;
    rhs(1, 0) = 3.0;
    rhs(1, 1) = 4.0;
    rhs(2, 0) = 5.0;
    rhs(2, 1) = 6.0;

    rhs.Print("rhs");

    auto AT = A.Transpose();
    AT.PrintDense("AT:");

    auto ab = A.Mult(rhs);
    ab.Print("ab:");

    auto ab_T = A.MultCT(rhs);
    ab_T.Print("ab_T:");

    auto ba = A.MultAT(rhs);
    auto ba_at = AT.Mult(rhs);

    ba.Print("ba:");
    ba_at.Print("ba from AT:");

    SparseMatrix<double> B;
    B = A;

    SparseMatrix<double> B2(A);

    auto C = A.Mult(B);
    C.PrintDense("C:");

    auto C2 = A.ToDense().Mult(B.ToDense());
    C2.Print("C dense:");


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

    // Test Mult Vector
    {
        Vector<double> x(size, 1.0);
        Vector<double> y(size);
        A.PrintDense("A:");
        A.Mult(x, y);
        std::cout << " x: " << x;
        std::cout << " Ax: " << y;

        A.MultAT(x, y);
        std::cout << " A^T x: " << y;
    }

    // Test Sort Indices
    {
        const int size = 2;
        const int nnz = 4;
        std::vector<int> indptr(size + 1);
        std::vector<int> indices(nnz);
        std::vector<double> data(nnz);

        indptr[0] = 0;
        indptr[1] = 2;
        indptr[2] = nnz;

        indices[0] = 1;
        indices[1] = 0;
        indices[2] = 1;
        indices[3] = 0;

        data[0] = 1;
        data[1] = 2;
        data[2] = 1;
        data[3] = 2;

        SparseMatrix<> A_sort(indptr, indices, data,
                              size, size);

        A_sort.PrintDense("A:");

        for (int i = 0; i < nnz; ++i)
        {
            printf("%d %.2f\n", A_sort.GetIndices()[i], A_sort.GetData()[i]);
        }

        A_sort.SortIndices();

        A_sort.PrintDense("A Sorted:");

        for (int i = 0; i < nnz; ++i)
        {
            printf("%d %.2f\n", A_sort.GetIndices()[i], A_sort.GetData()[i]);
        }
    }

    // Test Scalar operations
    {
        SparseMatrix<> A_scalar(A);
        A_scalar.PrintDense("A");

        A_scalar *= 2.0;
        A_scalar.PrintDense("A * 2.0");

        A_scalar /= 4.0;
        A_scalar.PrintDense("(A * 2.0) / 4");

        A_scalar = -1.0;
        A_scalar.PrintDense("A = -1");
    }
}

void test_coo()
{
    // Empty coo
    {
        CooMatrix<int> coo;
        SparseMatrix<int> sp_coo = coo.ToSparse();
        sp_coo.Print("Empty:");

        DenseMatrix dense_coo = coo.ToDense();
        dense_coo.Print("Empty:");
    }

    // Copy coo
    {
        CooMatrix<int> coo(3, 3);
        coo.AddSym(0, 0, 1);
        coo.AddSym(0, 1, 2);
        coo.AddSym(0, 2, 3);

        CooMatrix<int> coo2(coo);
        coo2.AddSym(0, 0, 1);

        CooMatrix<int> coo3;
        coo3 = coo2;
        coo3.AddSym(0, 0, 1);

        coo.Print("Coo 1:");
        coo2.Print("Coo 2:");
        coo3.Print("Coo 3:");
    }

    {
        constexpr int size = 1000;
        CooMatrix<int> coo_large(size);

        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; j += 2)
            {
                coo_large.AddSym(i, j, i * j);
            }
        }
    }

    // With setting specfic size
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

    // With symmetric add
    {
        CooMatrix<double> coo(10, 10);
        coo.AddSym(0, 0, 1.0);
        coo.AddSym(0, 1, 2.0);
        coo.AddSym(1, 1, 3.0);
        coo.AddSym(1, 1, 3.0);
        coo.AddSym(1, 1, 3.0);
        coo.AddSym(2, 2, 3.0);
        coo.AddSym(4, 2, 3.0);

        coo.ToDense().Print("Coo Symmetric Add");
    }
    // Make sure ToSparse gets same result as ToDense
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

        CooMatrix<double> coo2(coo);
        CooMatrix<double> coo3;
        coo3 = coo;

        SparseMatrix<int> sp = coo.ToSparse<int>();
    }

    // Generate larger coordinate matrix
    {
        const int size = 1e3;
        const int num_entries = 1e4;

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

    // With Mult
    {
        const int size = 10;

        CooMatrix<double> coo(size);
        coo.Add(0, 0, 1.0);
        coo.Add(0, 1, 2.0);
        coo.Add(1, 1, 3.0);
        coo.Add(1, 1, 3.0);
        coo.Add(1, 1, 3.0);
        coo.Add(2, 2, 3.0);
        coo.Add(4, 2, 3.0);
        coo.Add(8, 9, 3.0);

        Vector<double> x(size, 1.0);
        Vector<double> y(size);


        coo.ToDense().Print("coo:");
        std::cout << "x: " << x;

        coo.Mult(x, y);
        std::cout << "y: " << y;

        coo.MultAT(x, y);
        std::cout << "coo^T y: " << y;
    }

    // Eliminate zeros
    {
        CooMatrix<int> coo(3, 3);
        coo.AddSym(0, 0, 1);
        coo.AddSym(0, 1, 0);
        coo.AddSym(0, 1, 1e-10);
        coo.AddSym(0, 2, 3);

        coo.Print("Coo with zero:");

        coo.EliminateZeros();
        coo.Print("Coo no zero:");

        const double tolerance = 1e-8;
        coo.EliminateZeros(tolerance);
        coo.Print("Coo no really small:");
    }

}

void test_dense()
{
    const int size = 5;

    DenseMatrix d1;
    DenseMatrix d2(size);
    DenseMatrix d3(size, size);
    DenseMatrix d4(d3);
    DenseMatrix d5;
    d5 = d3;


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

    Vector<double> x(size, 1.0);
    Vector<double> y(size);

    d2.Mult(x, y);

    printf("d2 * x = y:\n");
    //std::cout << y;

    // printf("d2 * y:\n");
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
    // B.Print("B:");

    DenseMatrix C = A.Mult(B);

    // C.Print("C:");

    DenseMatrix D = A.MultAT(C);
    // D.Print("D:");

    DenseMatrix E = C.MultBT(B);
    // E.Print("E:");

    DenseMatrix F = B.MultABT(A);
    // F.Print("F:");

    F *= 2.0;
    // F.Print("2F:");
    F /= 2.0;
    // F.Print("F:");

    DenseMatrix G = 5 * F;
    DenseMatrix G2 = F * 5;
    // G.Print("5 *F:");
    // G2.Print("F *5:");

    Vector<double> v1(size);
    Vector<double> v2(size, 1.0);

    auto v3 = d2.Mult(v2);
    d2.Print("d2");
    v2.Print("v2");
    v3.Print("d2 * v2");

    auto v4 = d2.MultAT(v2);
    d2.Print("d2");
    v2.Print("v2");
    v4.Print("d2^T * v2");

    Vector<double> row_1 = d2.GetRow(1);
    Vector<double> col_1 = d2.GetCol(1);
    VectorView<double> col_view = d2.GetColView(1);

    row_1.Print("Row 1 of d2");
    col_1.Print("Col 1 of d2");
    col_view.Print("Col view of Col 1 of d2");

    col_view[0] = -20.0;
    d2.Print("d2 modify through view");

    row_1 = 5.0;
    col_1 = 5.0;

    d2.SetRow(1, row_1);
    d2.SetCol(1, col_1);

    d2.Print("d2 with row and col 1 set to 5.0");

    DenseMatrix rows_0_1 = d2.GetRow(1, 3);
    rows_0_1.Print("Rows 1,2 of d2");

    DenseMatrix cols_0_1 = d2.GetCol(1, 3);
    cols_0_1.Print("Cols 1,2 of d2");

    rows_0_1 = -0.5;
    d2.SetRow(3, rows_0_1);
    d2.Print("d2 with rows 3,4 set to -0.5");

    cols_0_1 = -9.5;
    d2.SetCol(3, cols_0_1);
    d2.Print("d2 with cols 3,4 set to -9.5");

    DenseMatrix submat = d2.GetSubMatrix(1, 1, 4, 4);
    submat.Print("(1,1) : (3,3) submatrix of d2");

    submat = 100;
    d2.SetSubMatrix(2, 2, 5, 5, submat);
    d2.Print("d2 with submatrix (2,2):(4,4) set to 100");

    DenseMatrix d2_T = d2.Transpose();
    d2_T.Print("d2^T:");

    const DenseMatrix d2_const = d2;
    d2_const.Print("d2 const:");
    const auto& d2_const_view = d2_const.GetColView(0);
    std::cout << "d2[0] "  << d2_const_view[0] << "\n";
    d2_const.GetColView(0).Print("D2 Col 0");

    // d2_const_view[0] = 1.0; // Fails correctly
}

void test_vector()
{
    const int size = 5;

    Vector<double> v1;
    Vector<double> v2(size);
    Vector<double> v3(size, 3.0);
    Vector<double> v3_copy(v3);
    Vector<double> v3_equal;
    v3_equal = v3;

    std::cout << "v1:" << v1;
    std::cout << "v2:" << v2;
    std::cout << "v3:" << v3;

    Normalize(v3);
    assert(std::fabs(L2Norm(v3) - 1.0) < 1e-10);

    std::cout << "v3 normalized:" << v3;

    std::cout << "v3_copy:" << v3_copy;
    std::cout << "v3_equal:" << v3_equal;

    std::cout << "v3_copy == v3_equal: ";
    std::cout << std::boolalpha << (v3_copy == v3_equal) << "\n";

    std::cout << "v3_copy == v3 normalized: ";
    std::cout << std::boolalpha << (v3_copy == v3) << "\n";

    std::cout << "v3[0]: " << v3[0] << "\n";

    auto v3v3 = v3 * v3;
    std::cout << "v3 * v3: " << v3v3 << "\n";

    const double alpha = 3;
    const double beta = 5.1;

    std::cout << "v3 *= 3: " << (v3 *= alpha);
    std::cout << "v3 /= 5.1: " << (v3 /= beta);

    std::cout << "3 * v3: " << alpha* v3;
    std::cout << "v3 * 3: " << v3* alpha;

    std::cout << "v3 * 5.1" << v3* beta;
    std::cout << "5.1 * v3" << beta* v3;

    std::cout << "v3 / 5.1" << v3 / beta;
    std::cout << "5.1 / v3" << beta / v3;

    std::cout << "v3 = 3" << (v3 = alpha);
    std::cout << "v3 = 5" << (v3 = beta);

    std::iota(std::begin(v3), std::end(v3), 0);
    std::cout << "iota(v3)" << v3;

    std::cout << "v3 - (1.5 * v3_copy): " << v3.Sub(1.5, v3_copy);
    std::cout << "v3 + (0.5 * v3_copy): " << v3.Add(0.5, v3_copy);

    v3.Add(alpha, v3_copy).Sub(beta, v3_equal);
    std::cout << "v3 + (alpha * v3_copy) - (beta * v3_equal): " << v3;

    // Entry-wise operators
    std::cout << "(v3)_i * (v3_copy)_i: " << (v3 *= v3_copy);
    std::cout << "(v3)_i / (v3_copy)_i: " << (v3 /= v3_copy);

    std::cout << "(v3) += 3" << (v3 += alpha);
    std::cout << "(v3) -= 5.1" << (v3 -= beta);

    // Remove constant vector v := Mean(v3)
    SubAvg(v3);
    std::cout << "SubAvg(v3)" << v3;

    // Vector Stats
    std::cout << "Max(v3):    " << Max(v3) << "\n";
    std::cout << "Min(v3):    " << Min(v3) << "\n";

    std::cout << "AbsMax(v3): " << AbsMax(v3) << "\n";
    std::cout << "AbsMin(v3): " << AbsMin(v3) << "\n";

    std::cout << "Sum(v3):    " << Sum(v3) << "\n";
    std::cout << "Mean(v3):   " << Mean(v3) << "\n";

    std::cout << "Sum(v3_copy):    " << Sum(v3_copy) << "\n";
    std::cout << "Mean(v3_copy):    " << Mean(v3_copy) << "\n";

}

void test_parser()
{
    // Write Vector
    std::vector<double> vect_out({1.0, 2.0, 3.0});
    WriteText<double>(vect_out, "vect.vect");

    // Write Integer Vector
    std::vector<int> vect_out_int({1, 2, 3});
    WriteText<int>(vect_out_int, "vect_int.vect");

    CooMatrix<double> coo_out(3, 3);
    coo_out.Add(0, 0, 1.0);
    coo_out.Add(1, 1, 2.0);
    coo_out.Add(1, 2, 2.0);
    coo_out.Add(2, 0, 3.0);
    coo_out.Add(2, 2, 3.0);

    SparseMatrix<double> sp_out = coo_out.ToSparse();

    // Write Adjacency List
    WriteAdjList(sp_out, "adj.adj");

    // Write Coordinate List
    WriteCooList(sp_out, "coo.coo");

    // Read Vector
    std::vector<double> vect = ReadText("vect.vect");
    Vector<double> v(vect);
    v.Print("vect:");

    // Read Integer Vector
    std::vector<int> vect_i = ReadText<int>("vect_int.vect");
    Vector<int> v_i(vect_i);
    v_i.Print("vect:");

    // Read List formats
    SparseMatrix<double> adj = ReadAdjList("adj.adj");
    SparseMatrix<double> coo = ReadCooList("coo.coo");

    adj.PrintDense("Adj:");
    coo.PrintDense("Coo:");

    // Symmetric file type
    bool symmetric = true;
    SparseMatrix<double> adj_sym = ReadAdjList("adj.adj", symmetric);
    SparseMatrix<double> coo_sym = ReadCooList("coo.coo", symmetric);
    adj_sym.PrintDense("Adj Sym:");
    coo_sym.PrintDense("Coo Sym:");

    // Integer file type
    SparseMatrix<int> adj_int = ReadAdjList<int>("adj.adj");
    SparseMatrix<int> coo_int = ReadCooList<int>("coo.coo");

    adj_int.PrintDense("Adj int:");
    coo_int.PrintDense("Coo int:");

    // Test non-existant file
    try
    {
        SparseMatrix<int> coo_int = ReadCooList<int>("fake.fake");
    }
    catch (std::runtime_error e)
    {
        printf("%s\n", e.what());
    }

    // Read AdjList as Integer Vector
    std::vector<int> vect_adj = ReadText<int>("adj.adj");
    Vector<int> v_adj(vect_adj);
    v_adj.Print("adjlist vector:");

    // Write Table to file
    CooMatrix<int> coo_table;
    coo_table.Add(0, 0, 1);
    coo_table.Add(0, 1, 1);
    coo_table.Add(0, 2, 1);
    coo_table.Add(1, 2, 1);
    coo_table.Add(2, 1, 1);
    coo_table.Add(2, 0, 1);

    SparseMatrix<int> sp_table = coo_table.ToSparse();
    WriteTable(sp_table, "table.table");
    sp_table.PrintDense("Table Write");

    // Read Table from file
    SparseMatrix<int> sp_table2 = ReadTable("table.table");
    sp_table2.PrintDense("Table Read");

    /*
        SparseMatrix<int> elem_node = ReadTable("element_node.txt");
        SparseMatrix<int> node_elem = elem_node.Transpose();

        SparseMatrix<int> elem_elem = elem_node.Mult(node_elem);
        SparseMatrix<int> node_node = node_elem.Mult(elem_node);

        printf("Elem Node: %d %d\n", elem_node.Rows(), elem_node.Cols());
        printf("Node Elem: %d %d\n", node_elem.Rows(), node_elem.Cols());
        printf("Elem Elem: %d %d\n", elem_elem.Rows(), elem_elem.Cols());
        printf("Node Node: %d %d\n", node_node.Rows(), node_node.Cols());
        node_node.Print("node node");
    */

    {
        std::vector<double> out_vect(10, 1.0);
        WriteBinary(out_vect, "vect.bin");

        std::vector<double> in_vect = ReadBinaryVect("vect.bin");

        std::cout << "out vect: " << out_vect;
        std::cout << "in vect: " << in_vect;

        WriteBinary(sp_out, "mat.bin");

        auto sp_in = ReadBinaryMat("mat.bin");

        sp_out.Print("Mat out:");
        sp_out.PrintDense("Mat  out:");
        sp_in.Print("Mat In:");
        sp_in.PrintDense("Mat In:");
    }


}

void test_operator()
{
    auto mult = [](const Operator & op)
    {
        Vector<double> vect(op.Cols(), 1);
        Vector<double> vect2(op.Rows(), 0);

        Randomize(vect);
        op.Mult(vect, vect2);

        return vect2;
    };

    auto multAT = [](const Operator & op)
    {
        Vector<double> vect(op.Cols(), 1);
        Vector<double> vect2(op.Rows(), 0);

        Randomize(vect);
        op.MultAT(vect, vect2);

        return vect2;
    };

    CooMatrix<double> coo(3, 3);
    coo.Add(0, 0, 1.0);
    coo.Add(0, 1, -2.0);
    coo.Add(1, 1, 2.0);
    coo.Add(1, 0, -3.0);
    coo.Add(2, 2, 4.0);

    SparseMatrix<double> sparse = coo.ToSparse();
    DenseMatrix dense = coo.ToDense();

    auto vect_coo = mult(coo);
    auto vect_dense = mult(dense);
    auto vect_sparse = mult(sparse);

    std::cout << "vect_coo" << vect_coo;
    std::cout << "vect_dense" << vect_dense;
    std::cout << "vect_sparse" << vect_sparse;

    auto vect_sparse_T = multAT(sparse);
    auto vect_coo_T = multAT(coo);
    auto vect_dense_T = multAT(dense);

    std::cout << "vect_coo_T" << vect_coo_T;
    std::cout << "vect_dense_T" << vect_dense_T;
    std::cout << "vect_sparse_T" << vect_sparse_T;
}

void test_solvers()
{
    const int size = 5;

    CooMatrix<double> coo(size, size);

    for (int i = 0; i < size - 1; ++i)
    {
        coo.AddSym(i, i, 2.0);
        coo.AddSym(i, i + 1, -1.0);
    }

    coo.AddSym(size - 1, size - 1, 2.0);

    SparseMatrix<double> A = coo.ToSparse();

    Vector<double> b(A.Cols(), 1.0);
    // Randomize(b);
    // Normalize(b);

    int max_iter = size;
    double rel_tol = 1e-12;
    double abs_tol = 1e-16;
    bool verbose = true;

    Vector<double> x = CG(A, b, max_iter, rel_tol, abs_tol, verbose);
    Vector<double> x_coo = CG(coo, b, max_iter, rel_tol, abs_tol, verbose);

    Vector<double> Ax = A.Mult(x);
    Vector<double> res = b - Ax;
    double error = L2Norm(res);

    if (size < 10)
    {
        A.PrintDense("A:");
        b.Print("b:");
        x.Print("x:");
        x_coo.Print("x_coo:");
        Ax.Print("Ax:");
    }

    printf("CG error: %.2e\n", error);

    std::vector<double> diag(size, 0.5);
    SparseMatrix<double> M(diag);

    Vector<double> px = PCG(A, M, b, max_iter, rel_tol, abs_tol, verbose);
    Vector<double> pAx = A.Mult(px);
    Vector<double> pres = b - pAx;
    double perror = L2Norm(pres);

    printf("PCG error: %.2e\n", perror);

    Vector<double> mx = MINRES(A, b, max_iter, rel_tol, abs_tol, verbose);
    Vector<double> mAx = A.Mult(mx);
    Vector<double> mres = b - mAx;
    double merror = L2Norm(mres);

    printf("MINRES error: %.2e\n", merror);

    Vector<double> pmx = PMINRES(A, M, b, max_iter, rel_tol, abs_tol, verbose);
    Vector<double> pmAx = A.Mult(pmx);
    Vector<double> pmres = b - pmAx;
    double pmerror = L2Norm(pmres);

    printf("pMINRES error: %.2e\n", pmerror);

    std::cout << pmx;
}

void test_blockmatrix()
{
    std::vector<int> row_offsets{0, 2, 4};
    std::vector<int> col_offsets{0, 2, 4};

    BlockMatrix<double> A;
    BlockMatrix<double> A2(row_offsets);
    BlockMatrix<double> A3(row_offsets, col_offsets);

    CooMatrix<double> coo(2, 2);
    coo.Add(0, 0, 1.0);
    coo.Add(1, 1, 2.0);

    SparseMatrix<double> block = coo.ToSparse();

    A2.GetBlock(0, 0).PrintDense("A(0,0)");

    A2.Print("A");
    A2.PrintDense("A Dense");

    A2.SetBlock(0, 0, block);
    A2.SetBlock(0, 1, block);
    A2.SetBlock(1, 0, block);
    A2.SetBlock(1, 1, block);

    A2.Print("A");
    A2.PrintDense("A Dense");

    A2.GetBlock(0, 0).PrintDense("A(0,0)");

    A2.Combine().PrintDense("A combined");

    Vector<double> x(A2.Cols());
    Vector<double> y(A2.Cols());

    //x[0] = 0;
    //x[1] = 1;
    //x[2] = 2;
    //x[3] = 3;

    Randomize(x);
    Randomize(y);

    A2.Mult(x, y);

    y.Print("y = Ax");

    A2.MultAT(x, y);

    y.Print("y = ATx");

    Randomize(x);
    Randomize(y);
    auto Ax = A2.Mult(x);
    auto Ay = A2.Mult(y);

    auto yAx = InnerProduct(y, Ax);
    auto xAy = InnerProduct(x, Ay);

    printf("%.8f %.8f\n", y.Mult(Ax), x.Mult(Ay));
    printf("%.8f %.8f\n", yAx, xAy);
}

void test_blockvector()
{
    CooMatrix<double> coo(4, 4);
    coo.Add(0, 0, 1.0);
    coo.Add(1, 1, 2.0);
    coo.Add(2, 2, 3.0);
    coo.Add(3, 3, 4.0);
    auto sparse = coo.ToSparse();

    std::vector<int> offsets{0, 2, 4};

    BlockVector<double> vect_empty;
    BlockVector<double> vect(offsets);

    Randomize(vect);

    vect_empty.Print("Vect Empty:");
    vect.Print("Vect:");

    VectorView<double> view0 = vect.GetBlock(0);
    VectorView<double> view1 = vect.GetBlock(1);
    // VectorView<double> view2 = vect.GetBlock(2); // Fails correctly

    view0.Print("Block 0");
    view1.Print("Block 1");

    printf("V0 * V1: %.8f\n", view0.Mult(view1));

    Normalize(view0);
    Normalize(view1);

    vect.Print("Vect Block Normalized through view:");

    view0[0] = -100.0;
    view1[0] = -200.0;

    vect.Print("Vect Modified through view:");


    {
        Vector<double> vect(offsets.back());

        vect[0] = 1.0;
        vect[1] = 2.0;
        vect[2] = 3.0;
        vect[3] = 4.0;

        const BlockVector<double> vect_const(vect, offsets);
        auto test_const = [] (const VectorView<double>& test)
        {
            test.Print("test");
        };

        test_const(vect_const.GetBlock(0));
        vect_const.GetBlock(0).Print("rvalue works ?");
        const VectorView<double>& test0 = vect_const.GetBlock(0);
        const VectorView<double>& test1 = vect_const.GetBlock(1);

        test0.Print("test0:");
        test1.Print("test1:");

        // test0[0] = 1.0; // Fails correctly

        test_const(test1);
    }

}

void test_blockoperator()
{
    CooMatrix<double> coo(2, 2);
    coo.Add(0, 0, 1.0);
    coo.Add(1, 1, 1.0);
    auto sparse = coo.ToSparse();
    auto dense = coo.ToDense();

    std::vector<int> offsets{0, 2, 4};

    Vector<double> x(offsets.back(), 1.0);
    Vector<double> y(offsets.back(), 0.0);

    {
        BlockOperator b;
    }

    {
        BlockOperator b(offsets);
        b.Mult(x, y);
        y.Print("y:");
    }
    {
        BlockOperator b(offsets, offsets);
        b.Mult(x, y);
        y.Print("y:");
    }
    // Symmetric
    {
        BlockOperator b(offsets, offsets);
        b.SetBlock(0, 0, sparse);
        b.SetBlock(0, 1, dense);
        b.SetBlock(1, 0, dense);
        b.SetBlock(1, 1, coo);

        b.Mult(x, y);
        y.Print("y:");

        b.MultAT(x, y);
        y.Print("y T:");
    }

    // Not Symmetric
    {
        BlockOperator b(offsets, offsets);
        b.SetBlock(0, 0, coo);
        b.SetBlock(0, 1, sparse);
        b.SetBlock(1, 1, dense);

        b.Mult(x, y);
        y.Print("y:");

        b.MultAT(x, y);
        y.Print("y T:");
    }

}

void test_timer()
{
    Timer timer;

    timer.Click(); // Start timer

    volatile double counter;

    for (int i = 0; i < 5000; ++i)
    {
        counter *= 1.0001;
    }

    timer.Click();  // 1 time step

    for (int i = 0; i < 50000; ++i)
    {
        // Do Hard Work
        counter *= 1.0001;
    }

    timer.Click(); // 2 time step

    for (int i = 0; i < 500000; ++i)
    {
        // Do Hard Work
        counter *= 1.0001;
    }

    timer.Click(); // 3 time step

    std::cout << "Time 0: " << timer[0] << std::endl;
    std::cout << "Time 1: " << timer[1] << std::endl;
    std::cout << "Time 2: " << timer[2] << std::endl;
    // std::cout << "Time 3: " << timer[3] << std::endl; // Correctly fails assertion
    std::cout << "Time Total: " << timer.TotalTime() << std::endl;
}

void test_eigensolve()
{
    DenseMatrix A(3, 3);

    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;

    A.Print("Eigen Input:");

    EigenSolver eigen;
    EigenSolver eigen2;
    eigen2 = eigen;

    auto eigen_pair = eigen.Solve(A, 1.0, 3);

    EigenSolver eigen3(eigen);

    std::cout << "EigenValues:" << eigen_pair.first << "\n";
    eigen_pair.second.Print("Eigenvectors:");

    DenseMatrix A2(3, 3);

    A2(0, 0) = 1.0;
    A2(0, 1) = -1.0;
    A2(1, 0) = -1.0;
    A2(1, 1) = 2.0;
    A2(1, 2) = -1.0;
    A2(2, 1) = -1.0;
    A2(2, 2) = 1.0;

    A2.Print("Eigen Input:");

    auto eigen_pair2 = eigen3.Solve(A2, 0.5, 3);

    std::cout << "EigenValues:" << eigen_pair2.first << "\n";
    eigen_pair2.second.Print("Eigenvectors:");

    assert(eigen_pair2.second.Cols() == 2);

    assert(std::fabs(eigen_pair2.first[0]) < 1e-15);
    assert(std::fabs(eigen_pair2.first[1] - 1.0) < 1e-15);
    assert(std::fabs(eigen_pair2.first[2] - 3.0) < 1e-15);
}

void test_argparser(int argc, char** argv)
{
    if (argc > 1)
    {
        ArgParser arg_parser(argc, argv);
    }

    // Good Parser
    {
        const int test_argc = 7;
        const char* test_argv[test_argc];

        test_argv[0] = "test";
        test_argv[1] = "-bf";
        test_argv[2] = "-bt";
        test_argv[3] = "-dt";
        test_argv[4] = "5";
        test_argv[5] = "-st";
        test_argv[6] = "StringTest";

        ArgParser arg_parser(test_argc, test_argv);

        bool test_bool = true;
        bool test_bool_f = false;
        bool test_bool_t = true;
        double test_double = -1.0;
        std::string test_string = "default_string";
        std::string test_string_default = "default_string";

        arg_parser.Parse(test_bool, "-b", "Bool No Change Test");
        arg_parser.Parse(test_bool_f, "-bf", "Bool False Test");
        arg_parser.Parse(test_bool_t, "-bt", "Bool True Test");
        arg_parser.Parse(test_double, "-dt", "Double Test");
        arg_parser.Parse(test_string, "-st", "String Test");

        std::cout << "Good Parse!\n";

        if (!arg_parser.IsGood())
        {
            arg_parser.ShowHelp();
            arg_parser.ShowErrors();

            assert(false);
        }

        assert(test_bool == true);
        assert(test_bool_f == true);
        assert(test_bool_t == false);
        assert(test_double = 5.0);
        assert(test_string.compare("StringTest") == 0);
        assert(test_string_default.compare("default_string") == 0);

        arg_parser.ShowOptions();
    }

    // Bad Parser
    {
        const int test_argc = 6;
        const char* test_argv[test_argc];

        test_argv[0] = "test";
        test_argv[1] = "-bf"; // These two flags are the same,
        test_argv[2] = "-bf"; // a bad input
        test_argv[3] = "-bt";
        test_argv[4] = "-dt";
        test_argv[5] = "5";

        ArgParser arg_parser(test_argc, test_argv);

        bool test_bool = true;
        bool test_bool_f = false;
        bool test_bool_t = true;
        double test_double = -1.0;

        // These two share same flag
        arg_parser.Parse(test_bool, "-b", "First Bool No Change Test");
        arg_parser.Parse(test_bool, "-b", "Second Bool No Change Test");

        arg_parser.Parse(test_bool_f, "-bf", "Bool False Test");
        arg_parser.Parse(test_bool_t, "-bt", "Bool True Test");
        arg_parser.Parse(test_double, "-dt", "Double Test");

        if (!arg_parser.IsGood())
        {
            std::cout << "Bad Parse:\n";
            arg_parser.ShowHelp();
            arg_parser.ShowErrors();
        }
        else
        {
            assert(false);
        }

        arg_parser.ShowOptions();
    }

    // Includes help
    {
        const int test_argc = 6;
        const char* test_argv[test_argc];

        test_argv[0] = "test";
        test_argv[1] = "-bf";
        test_argv[2] = "-bt";
        test_argv[3] = "-dt";
        test_argv[4] = "5";
        test_argv[5] = "--help";

        ArgParser arg_parser(test_argc, test_argv);

        bool test_bool = true;
        bool test_bool_f = false;
        bool test_bool_t = true;
        double test_double = -1.0;

        arg_parser.Parse(test_bool, "-b", "Bool No Change Test");
        arg_parser.Parse(test_bool_f, "-bf", "Bool False Test");
        arg_parser.Parse(test_bool_t, "-bt", "Bool True Test");
        arg_parser.Parse(test_double, "-dt", "Double Test");

        if (!arg_parser.IsGood())
        {
            std::cout << "Help Parse:\n";
            arg_parser.ShowHelp();
            arg_parser.ShowErrors();
        }
        else
        {
            assert(false);
        }

        arg_parser.ShowOptions();
    }
}

int main(int argc, char** argv)
{
    test_coo();
    test_vector();
    test_sparse();
    test_operator();
    test_solvers();
    test_lil();
    test_dense();
    test_blockmatrix();
    test_blockvector();
    test_blockoperator();
    test_parser();
    test_timer();
    test_eigensolve();
    test_argparser(argc, argv);

    return EXIT_SUCCESS;
}
