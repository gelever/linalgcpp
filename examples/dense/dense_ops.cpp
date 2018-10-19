#include "../include/ex_utilities.hpp"

DenseMatrix parse_dense(const std::string& filename)
{
    return linalgcpp::ReadCooList(filename).ToDense();
}

void constructors()
{
    DenseMatrix d_0;
    DenseMatrix d_2_2(2);
    DenseMatrix d_2_3(2, 3);

    d_0.Print("d 0x0");
    d_2_2.Print("d 2x2");
    d_2_3.Print("d 2x3");

    DenseMatrix from_file = parse_dense("data/mat2.txt");
    from_file.Print("From File:");
}

DenseMatrix transpose(const DenseMatrix& A)
{
    int A_rows = A.Rows();
    int A_cols = A.Cols();

    DenseMatrix AT(A_cols, A_rows);

    for (int i = 0; i < A_rows; ++i)
    {
        for (int j = 0; j < A_cols; ++j)
        {
            AT(j, i) = A(i, j);
        }
    }

    return AT;
}

Vector matvec(const DenseMatrix& A, const Vector& x)
{
    assert(x.size() == A.Cols());

    Vector b(A.Rows());

    int A_rows = A.Rows();
    int A_cols = A.Cols();

    for (int i = 0; i < A_rows; ++i)
    {
        double sum = 0.0;

        for (int j = 0; j < A_cols; ++j)
        {
            sum += A(i, j) * x[j];
        }

        b[i] = sum;
    }

    return b;
}

void matvec_ops()
{
    DenseMatrix m(4, 2);

    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(2, 0) = 3.0;
    m(3, 1) = 4.0;

    m.Print("m:");

    DenseMatrix m_T = transpose(m);
    m_T.Print("m_T:");

    Vector b_2(2);
    b_2[0] = -1.0; b_2[1] = 3.0;

    Vector b_4(4);
    b_4[0] = -1.5; b_4[1] = 2.5;
    b_4[2] = -1.5; b_4[3] = 3.5;

    Vector mb = matvec(m, b_2);
    Vector m_Tb = matvec(m_T, b_4);

    mb.Print("m * b_2");
    m_Tb.Print("m^T * b_4");
}

void access_ops()
{
    int dim = 4;

    DenseMatrix d_0 = parse_dense("data/mat4.txt");
    DenseMatrix d_1 = parse_dense("data/mat4.2.txt");

    d_0.Print("d_0", std::cout, 6, 1);
    d_1.Print("d_1", std::cout, 6, 1);

    std::cout << "d_0 size: " << d_0.Rows() << " x " << d_0.Cols() << "\n";
    std::cout << "d_1 size: " << d_1.Rows() << " x " << d_1.Cols() << "\n\n";

    std::cout << "d_0 (1, 2): " << d_0(1, 2) << "\n";
    std::cout << "d_1 (2, 3): " << d_1(2, 2) << "\n\n";

    DenseMatrix d0_plus_d1 = d_0 + d_1;
    DenseMatrix d0_sub_d1 = d_0 - d_1;

    d0_plus_d1.Print("d0 + d1");
    d0_sub_d1.Print("d0 - d1");

    DenseMatrix d1_x_10 = 10.0 * d_1;
    d1_x_10.Print("10.0 * d0");

    d1_x_10 /= 4.0;
    d1_x_10.Print("(10.0 * d0) / 4.0");
}

DenseMatrix mat_mult_mat(const DenseMatrix& A, const DenseMatrix& B)
{
    DenseMatrix C(A.Rows(), B.Cols());

    for (int k = 0; k < B.Cols(); ++k)
    {
        for (int i = 0; i < A.Rows(); ++i)
        {
            double sum = 0.0;

            for (int j = 0; j < B.Rows(); ++j)
            {
                sum += A(i, j) * B(j, k);
            }

            C(i, k) = sum;
        }
    }

    return C;
}

void matmat_ops()
{
    DenseMatrix m(4, 2);

    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(2, 0) = 3.0; m(3, 1) = 4.0;

    DenseMatrix m_T = transpose(m);

    m.Print("m:");
    m_T.Print("m_T:");

    DenseMatrix m_T_m = mat_mult_mat(m_T, m);
    m_T_m.Print("m^T * m");

    DenseMatrix m_T_m_blas = m_T.Mult(m);
    m_T_m_blas.Print("reference: m^T * m");

    // Other way around
    {
        DenseMatrix m_T_m = mat_mult_mat(m, m_T);
        m_T_m.Print("m * m^T");

        DenseMatrix m_T_m_blas = m.Mult(m_T);
        m_T_m_blas.Print("reference: m * m^T");
    }
}


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

DenseMatrix random_mat(int n, int m)
{
    std::vector<double> data(n * m);

    std::random_device r;
    std::default_random_engine dev(r());
    std::uniform_real_distribution<double> gen(-1.0, 1.0);

    for (double& val : data)
    {
        val = gen(dev);
    }

    return DenseMatrix(n, m, std::move(data));
}

double test_LU(int n)
{
    DenseMatrix A = random_mat(n, n);

    Timer timer(Timer::Start::True);
    auto lu = lu_decomp(A);
    timer.Click();

    auto& L = lu.first;
    auto& U = lu.second;

    Vector b(A.Cols(), 1.0);
    Vector x = solve_lu(L, U, b);
    Vector Ax = A.Mult(x);
    Vector res = b - Ax;

    std::cout << "LU Relative Residual norm:" << res.L2Norm() / b.L2Norm() << "\n";

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

    b.Print("b:");
    x.Print("x:");
    Ax.Print("Ax:");
}

void time_LU()
{
    for (int i = 10; i < 2000; i *= 1.5)
    {
        std::cout << i << " LU \n";

        double lu_time = test_LU(i);

        std::cout << "Total Time: " << lu_time << "\n\n";
    }
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

    auto& Q = qr.first;
    auto& R = qr.second;

    Vector b(A.Rows(), 1.0);
    Vector x = solve_qr(Q, R, b);
    Vector Ax = A.Mult(x);
    Vector res = b - Ax;

    std::cout << "QR Relative Residual norm:" << res.L2Norm() / b.L2Norm() << "\n";

    return timer.TotalTime();
}

void time_QR()
{
    for (int i = 10; i < 2000; i *= 1.5)
    {
        int m = i;
        int n = i * 1.5;

        std::cout << m << " * " << n << " QR \n";

        double qr_time = test_QR(m, n);

        std::cout << "Total Time: " << qr_time << "\n\n";
    }
}

void QR()
{
    //DenseMatrix A = parse_dense("data/mat4.txt");
    //int n = 10;
    //DenseMatrix A = random_mat(n, n);
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

    b.Print("b:");
    x.Print("x:");
    Ax.Print("Ax:");
}

void QR_eig()
{
    //DenseMatrix A = parse_dense("data/mat4.txt");
    int n = 10;
    DenseMatrix A = random_mat(n, n);

    for (int i = 0; i < A.Rows(); ++i)
    {
        for (int j = i; j < A.Cols(); ++j)
        {
            A(i, j) = A(j, i);
        }
    }

    //Reference
    linalgcpp::EigenSolver es;
    auto eig_pair = es.Solve(A, 1.0, A.Rows());

    DenseMatrix U(A.Rows());
    DenseMatrix buffer(A.Rows());

    for (int i = 0; i < U.Rows(); ++i)
    {
        U(i, i) = 1.0;
    }

    for (int i = 0; i < 100000; ++i)
    {
        auto qr = qr_decomp(A);
        auto& Q = qr.first;
        auto& R = qr.second;

        R.Mult(Q, A);
        U.Mult(Q, buffer);
        swap(U, buffer);
    }

    A.Print("T:");
    U.Print("U:");

    using linalgcpp::operator<<;
    std::cout << "Ref Eigs: " << eig_pair.first;
    eig_pair.second.Print("Ref Evects");

}

void ops()
{
    //access_ops();
    //matvec_ops();
    //matmat_ops();

    //LU();
    //time_LU();

    //QR();
    //time_QR();
    QR_eig();

}

int main(int argc, char** argv)
{
    constructors();
    ops();

    return EXIT_SUCCESS;
}
