#include "../include/ex_utilities.hpp"

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

Vector solve_qr(const DenseMatrix& Q, const DenseMatrix& R, const Vector& b)
{
    Vector QT_b = Q.MultAT(b);
    Vector x = back_solve(R, QT_b);

    return x;
}


std::pair<DenseMatrix, DenseMatrix>
qr_decomp(const DenseMatrix& A)
{
    int m = A.Rows();
    int n = A.Cols();

    DenseMatrix Q(A);
    DenseMatrix R(n, n);

    for (int i = 0; i < n; ++i)
    {
        const auto&& a_i = A.GetColView(i);
        auto q_i = Q.GetColView(i);

        for (int j = 0; j < i; ++j)
        {
            auto q_j = Q.GetColView(j);

            double ae_i = a_i.Mult(q_j);

            R(j, i) = ae_i;

            linalgcpp::Add(1.0, q_i, -ae_i, q_j, 0.0, q_i);
        }

        q_i /= q_i.L2Norm();
        R(i, i) = a_i.Mult(q_i);
    }

    return {Q, R};
}

double test_QR(int m, int n)
{
    DenseMatrix A = random_mat(m, n);

    Timer timer(Timer::Start::True);
    auto qr = qr_decomp(A);
    timer.Click();

    // Check residual
    // auto& Q = qr.first;
    // auto& R = qr.second;

    // Vector b(A.Rows(), 1.0);
    // Vector x = solve_qr(Q, R, b);
    // Vector Ax = A.Mult(x);
    // Vector res = b - Ax;

    // std::cout << "QR Relative Residual norm:" << res.L2Norm() / b.L2Norm() << "\n";

    return timer.TotalTime();
}

void time_QR()
{
    for (int i = 10; i < 2000; i *= 1.5)
    {
        int m = i;
        int n = i * 1.5;

        std::cout << "QR " << m << " * " << n << "\t";

        double qr_time = test_QR(m, n);

        std::cout << "Time: " << qr_time << "\n";
    }
}

void QR()
{
    DenseMatrix A(3, 2);
    A(0, 0) = 3;
    A(0, 1) = -6;
    A(1, 0) = 4;
    A(1, 1) = -8;
    A(2, 1) = 1;

    auto qr = qr_decomp(A);
    auto& Q = qr.first;
    auto& R = qr.second;

    A.Print("A:");
    Q.Print("Q:");
    R.Print("R:");

    Q.Mult(R).Print("QR:");

    auto Q_T = Q.Transpose();
    Q_T.Mult(Q).Print("QT * Q:");

    Vector b(A.Rows(), 1.0);
    b[0] = -1.0; b[1] = 7; b[2] = 2;
    Vector x = solve_qr(Q, R, b);
    Vector Ax = A.Mult(x);

    std::cout << "Least Squares Ax = b:\n";
    b.Print("b:");
    x.Print("x:");
    Ax.Print("Ax:");
}

void QR_eig()
{
    // Create random symmetric matrix
    int n = 8;
    DenseMatrix A = symmetrize(random_mat(n, n));

    //Reference eigenpairs
    linalgcpp::EigenSolver es;
    auto eig_pair = es.Solve(A, 1.0, A.Rows());

    DenseMatrix U(A.Rows());
    DenseMatrix buffer(A.Rows());

    // Initial U is identity
    for (int i = 0; i < U.Rows(); ++i)
    {
        U(i, i) = 1.0;
    }

    // Run QR
    for (int i = 0; i < 10000; ++i)
    {
        auto qr = qr_decomp(A);
        auto& Q = qr.first;
        auto& R = qr.second;

        R.Mult(Q, A);
        U.Mult(Q, buffer);
        swap(U, buffer);
    }

    // Show decomposition and reference solution
    std::cout << "QR eigen decomposition:\n";
    A.Print("T:");
    U.Print("U:");

    using linalgcpp::operator<<;
    std::cout << "Ref Eigs: " << eig_pair.first;
    eig_pair.second.Print("Ref Evects");

}

int main()
{
    QR();
    QR_eig();
    time_QR();
}
