#include "solvers.hpp"

namespace linalgcpp
{

Vector<double> CG(const Operator& A, const Vector<double>& b,
                  int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    CG(A, b, x, max_iter, tol, verbose);

    return x;
}

void CG(const Operator& A, const Vector<double>& b, Vector<double>& x,
        int max_iter, double tol, bool verbose)
{
    assert(A.Cols() == b.size());
    assert(A.Rows() == x.size());
    assert(A.Rows() == A.Cols());

    Vector<double> Ap = A.Mult(x);
    Vector<double> r = b - Ap;
    Vector<double> p = r;

    const double r0 = InnerProduct(r, r);
    const double tol_tol = r0 * tol * tol;

    for (int k = 0; k < max_iter; ++k)
    {
        A.Mult(p, Ap);

        double alpha = (r * r) / (p * Ap);

        x.Add(alpha, p);

        double denom = InnerProduct(r, r);

        r.Sub(alpha, Ap);

        double numer = InnerProduct(r, r);

        if (verbose)
        {
            printf("CG %d: %.2e\n", k, numer / r0);

        }

        if (numer < tol_tol)
        {
            break;
        }

        double beta = numer / denom;

        p *= beta;
        p += r;
    }
}

Vector<double> PCG(const Operator& A, const Operator& M, const Vector<double>& b,
                   int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    PCG(A, M, b, x, max_iter, tol, verbose);

    return x;
}

void PCG(const Operator& A, const Operator& M, const Vector<double>& b, Vector<double>& x,
         int max_iter, double tol, bool verbose)
{
    assert(A.Cols() == b.size());
    assert(A.Rows() == x.size());
    assert(A.Rows() == A.Cols());
    assert(M.Rows() == A.Rows());
    assert(M.Cols() == A.Cols());

    Vector<double> Ap = A.Mult(x);
    Vector<double> r = b - Ap;
    Vector<double> z = M.Mult(r);
    Vector<double> p = z;

    const double r0 = InnerProduct(z, r);
    const double abs_tol = 1e-24;
    const double tol_tol = std::max(r0 * tol * tol, abs_tol);

    for (int k = 0; k < max_iter; ++k)
    {
        A.Mult(p, Ap);

        double alpha = (r * z) / (p * Ap);

        x.Add(alpha, p);

        double denom = InnerProduct(z, r);

        r.Sub(alpha, Ap);
        M.Mult(r, z);

        double numer = InnerProduct(z, r);

        if (verbose)
        {
            printf("PCG %d: %.2e\n", k, numer / r0);
        }

        if (numer < tol_tol)
        {
            break;
        }

        double beta = numer / denom;

        p *= beta;
        p += z;
    }
}

Vector<double> MINRES(const Operator& A, const Vector<double>& b,
                  int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    MINRES(A, b, x, max_iter, tol, verbose);

    return x;
}

void MINRES(const Operator& A, const Vector<double>& b, Vector<double>& x,
        int max_iter, double tol, bool verbose)
{
    assert(A.Cols() == b.size());
    assert(A.Rows() == x.size());
    assert(A.Rows() == A.Cols());

    const int size = A.Cols();

    Vector<double> w0(size, 0.0);
    Vector<double> w1(size, 0.0);
    Vector<double> q(size, 0.0);
    Vector<double> v0(size, 0.0);
    Vector<double> v1 = b - A.Mult(x);

    double beta = v1.L2Norm();
    double eta = beta;

    double gamma = 1.0;
    double gamma2 = 1.0;

    double sigma = 0;
    double sigma2 = 0;

    for (int k = 0; k < max_iter; ++k)
    {
        v1 /= beta;
        A.Mult(v1, q);

        const double alpha = v1 * q;

        for (int i = 0; i < size; ++i)
        {
            v0[i] = q[i] - (beta * v0[i]) - (alpha * v1[i]);
        }

        const double delta = gamma2 * alpha - gamma * sigma2 * beta;
        const double rho3 = sigma * beta;
        const double rho2 = sigma2 * alpha + gamma * gamma2 * beta;

        beta = v0.L2Norm();

        const double rho1 = std::sqrt((delta * delta) + (beta * beta));

        for (int i = 0; i < size; ++i)
        {
            w0[i] = ((1.0 / rho1) * v1[i]) - ( (rho3/ rho1)  * w0[i]) - (( rho2 / rho1) * w1[i]);
        }

        gamma = gamma2;
        gamma2 = delta / rho1;

        for (int i = 0; i < size; ++i)
        {
            x[i] += gamma2 * eta * w0[i];
        }

        sigma = sigma2;
        sigma2 = beta / rho1;

        eta = -sigma2 * eta;

        if (verbose)
        {
            printf("MINRES %d: %.2e\n", k, eta);
        }

        if (std::fabs(eta) < tol)
        {
            break;
        }

        Swap(v0, v1);
        Swap(w0, w1);
    }
}

Vector<double> PMINRES(const Operator& A, const Operator& M, const Vector<double>& b,
                      int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    PMINRES(A, M, b, x, max_iter, tol, verbose);

    return x;

}

void PMINRES(const Operator& A, const Operator& M, const Vector<double>& b, Vector<double>& x,
            int max_iter, double tol, bool verbose)
{
    assert(A.Cols() == b.size());
    assert(A.Rows() == x.size());
    assert(A.Rows() == A.Cols());

    const int size = A.Cols();

    Vector<double> w0(size, 0.0);
    Vector<double> w1(size, 0.0);
    Vector<double> q(size, 0.0);
    Vector<double> v0(size, 0.0);
    Vector<double> v1 = b - A.Mult(x);
    Vector<double> u1 = M.Mult(v1);

    double beta = std::sqrt(InnerProduct(u1, v1));
    double eta = beta;

    double gamma = 1.0;
    double gamma2 = 1.0;

    double sigma = 0;
    double sigma2 = 0;

    for (int k = 0; k < max_iter; ++k)
    {
        v1 /= beta;
        u1 /= beta;

        A.Mult(u1, q);

        const double alpha = u1 * q;

        for (int i = 0; i < size; ++i)
        {
            v0[i] = q[i] - (beta * v0[i]) - (alpha * v1[i]);
        }

        const double delta = gamma2 * alpha - gamma * sigma2 * beta;
        const double rho3 = sigma * beta;
        const double rho2 = sigma2 * alpha + gamma * gamma2 * beta;

        M.Mult(v0, q);
        beta = std::sqrt(InnerProduct(v0, q));

        const double rho1 = std::sqrt((delta * delta) + (beta * beta));

        for (int i = 0; i < size; ++i)
        {
            w0[i] = ((1.0 / rho1) * u1[i]) - ( (rho3/ rho1)  * w0[i]) - (( rho2 / rho1) * w1[i]);
        }

        gamma = gamma2;
        gamma2 = delta / rho1;

        for (int i = 0; i < size; ++i)
        {
            x[i] += gamma2 * eta * w0[i];
        }

        sigma = sigma2;
        sigma2 = beta / rho1;

        eta = -sigma2 * eta;

        if (verbose)
        {
            printf("PMINRES %d: %.2e\n", k, eta);
        }

        if (std::fabs(eta) < tol)
        {
            break;
        }

        Swap(u1, q);
        Swap(v0, v1);
        Swap(w0, w1);
    }

}

} //namespace linalgcpp
