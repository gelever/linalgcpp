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

} //namespace linalgcpp
