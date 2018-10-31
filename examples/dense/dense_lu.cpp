#include "../include/ex_utilities.hpp"

std::pair<DenseMatrix, DenseMatrix>
lu_decomp(const DenseMatrix& A)
{
    int n = A.Rows();

    DenseMatrix L(n, n);
    DenseMatrix U(A);

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            double a_ji = U(j, i) / U(i, i);

            for (int k = i; k < n; ++k)
            {
                U(j, k) -= a_ji * U(i, k);
            }

            L(j, i) = a_ji;
        }

        L(i, i) = 1.0;
    }

    return {L, U};
}

Vector forward_elim(const DenseMatrix& L, const Vector& b)
{
    int n = L.Rows();

    Vector x(n);

    for (int i = 0; i < n; ++i)
    {
        double sum = b[i];

        for (int j = i - 1; j >= 0; --j)
        {
            sum -= L(i, j) * x[j];
        }

        x[i] = sum / L(i, i);
    }

    return x;
}

Vector back_solve(const DenseMatrix& U, const Vector& b)
{
    int n = U.Rows();

    Vector x(n);

    for (int i = n - 1; i >= 0; --i)
    {
        double sum = b[i];

        for (int j = i + 1; j < n; ++j)
        {
            sum -= U(i, j) * x[j];
        }

        x[i] = sum / U(i, i);
    }

    return x;
}

Vector solve_lu(const DenseMatrix& L, const DenseMatrix& U, const Vector& b)
{
    Vector L_inv_b = forward_elim(L, b);
    Vector x = back_solve(U, L_inv_b);

    return x;
}

double test_LU(int n)
{
    DenseMatrix A = random_mat(n, n);

    Timer timer(Timer::Start::True);
    auto lu = lu_decomp(A);
    timer.Click();

    // Check residual
    //auto& L = lu.first;
    //auto& U = lu.second;

    // Vector b(A.Cols(), 1.0);
    // Vector x = solve_lu(L, U, b);
    // Vector Ax = A.Mult(x);
    // Vector res = b - Ax;

    //std::cout << "LU Relative Residual norm:" << res.L2Norm() / b.L2Norm() << "\n";

    return timer.TotalTime();
}

void LU()
{
    DenseMatrix A = parse_dense("data/mat4.txt");
    auto lu = lu_decomp(A);
    auto& L = lu.first;
    auto& U = lu.second;

    A.Print("A:");
    L.Print("L:");
    U.Print("U:");

    L.Mult(U).Print("LU:");

    Vector b(A.Cols(), 1.0);
    Vector x = solve_lu(L, U, b);
    Vector Ax = A.Mult(x);

    std::cout << "Solving Ax = b:\n";
    b.Print("b:");
    x.Print("x:");
    Ax.Print("Ax:");
}

void time_LU()
{
    for (int i = 10; i < 2000; i *= 1.5)
    {
        std::cout << "LU " << i << "\t";

        double lu_time = test_LU(i);

        std::cout << "Time: " << lu_time << "\n";
    }
}

int main()
{
    LU();
    time_LU();
}
