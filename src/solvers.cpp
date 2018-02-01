#include "solvers.hpp"

namespace linalgcpp
{

CGSolver::CGSolver(const Operator& A, int max_iter, double tol, bool verbose,
                   double (*Dot)(const VectorView<double>&, const VectorView<double>&))
    : A_(A), max_iter_(max_iter), tol_(tol), verbose_(verbose), Ap_(A.Rows()), r_(A.Rows()), p_(A.Rows()),
      Dot_(Dot)
{
    assert(A_.Rows() == A_.Cols());
}

size_t CGSolver::Rows() const
{
    return A_.Rows();
}

size_t CGSolver::Cols() const
{
    return A_.Cols();
}

void CGSolver::Mult(const VectorView<double>& b, VectorView<double>& x) const
{
    assert(x.size() == A_.Rows());
    assert(b.size() == A_.Rows());

    A_.Mult(x, Ap_);
    r_ = b;
    r_ -= Ap_;
    p_ = r_;

    const double r0 = (*Dot_)(r_, r_);
    const double tol_tol = r0 * tol_ * tol_;

    for (int k = 0; k < max_iter_; ++k)
    {
        A_.Mult(p_, Ap_);

        double alpha = (*Dot_)(r_, r_) / (*Dot_)(p_, Ap_);

        x.Add(alpha, p_);

        double denom = (*Dot_)(r_, r_);

        r_.Sub(alpha, Ap_);

        double numer = (*Dot_)(r_, r_);

        if (verbose_)
        {
            printf("CG %d: %.2e %.2e / %.2e\n", k, numer, numer / r0, tol_tol);
        }

        if (numer < tol_tol)
        {
            break;
        }

        double beta = numer / denom;

        p_ *= beta;
        p_ += r_;
    }
}

Vector<double> CG(const Operator& A, const VectorView<double>& b,
                  int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    CG(A, b, x, max_iter, tol, verbose);

    return x;
}

void CG(const Operator& A, const VectorView<double>& b, VectorView<double>& x,
        int max_iter, double tol, bool verbose)
{
    CGSolver cg(A, max_iter, tol, verbose);

    cg.Mult(b, x);
}

PCGSolver::PCGSolver(const Operator& A, const Operator& M, int max_iter, double tol, bool verbose)
    : A_(A), M_(M), max_iter_(max_iter), tol_(tol), verbose_(verbose), Ap_(A.Rows()), r_(A.Rows()), p_(A.Rows()),
      z_(A.Rows())
{
    assert(A_.Rows() == A_.Cols());
    assert(A_.Rows() == M_.Cols());
    assert(M_.Rows() == M_.Cols());
}

size_t PCGSolver::Rows() const
{
    return A_.Rows();
}

size_t PCGSolver::Cols() const
{
    return A_.Cols();
}

void PCGSolver::Mult(const VectorView<double>& b, VectorView<double>& x) const
{
    assert(x.size() == A_.Rows());
    assert(b.size() == A_.Rows());

    A_.Mult(x, Ap_);
    r_ = b;
    r_ -= Ap_;

    M_.Mult(r_, z_);
    p_ = z_;

    const double r0 = z_.Mult(r_);

    const double abs_tol = 1e-24;
    const double tol_tol = std::max(r0 * tol_ * tol_, abs_tol);

    for (int k = 0; k < max_iter_; ++k)
    {
        A_.Mult(p_, Ap_);

        double alpha = (r_ * z_) / (p_ * Ap_);

        x.Add(alpha, p_);

        double denom = z_.Mult(r_);

        r_.Sub(alpha, Ap_);
        M_.Mult(r_, z_);

        double numer = z_.Mult(r_);

        if (verbose_)
        {
            printf("PCG %d: %.2e\n", k, numer / r0);
        }

        if (numer < tol_tol)
        {
            break;
        }

        double beta = numer / denom;

        p_ *= beta;
        p_ += z_;
    }
}

Vector<double> PCG(const Operator& A, const Operator& M, const VectorView<double>& b,
                   int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    PCG(A, M, b, x, max_iter, tol, verbose);

    return x;
}

void PCG(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double>& x,
         int max_iter, double tol, bool verbose)
{
    PCGSolver pcg(A, M, max_iter, tol, verbose);

    pcg.Mult(b, x);
}

MINRESSolver::MINRESSolver(const Operator& A, int max_iter, double tol, bool verbose)
    : A_(A), max_iter_(max_iter), tol_(tol), verbose_(verbose),
      w0_(A.Rows()), w1_(A.Rows()),
      v0_(A.Rows()), v1_(A.Rows()),
      q_(A.Rows())
{
    assert(A_.Rows() == A_.Cols());
}

size_t MINRESSolver::Rows() const
{
    return A_.Rows();
}

size_t MINRESSolver::Cols() const
{
    return A_.Cols();
}

void MINRESSolver::Mult(const VectorView<double>& b, VectorView<double>& x) const
{
    assert(x.size() == A_.Rows());
    assert(b.size() == A_.Rows());

    const int size = A_.Cols();

    w0_ = 0.0;
    w1_ = 0.0;
    v0_ = 0.0;

    A_.Mult(x, q_);
    v1_ = b;
    v1_ -= q_;

    double beta = v1_.L2Norm();
    double eta = beta;

    double gamma = 1.0;
    double gamma2 = 1.0;

    double sigma = 0;
    double sigma2 = 0;

    for (int k = 0; k < max_iter_; ++k)
    {
        v1_ /= beta;
        A_.Mult(v1_, q_);

        const double alpha = v1_.Mult(q_);

        for (int i = 0; i < size; ++i)
        {
            v0_[i] = q_[i] - (beta * v0_[i]) - (alpha * v1_[i]);
        }

        const double delta = gamma2 * alpha - gamma * sigma2 * beta;
        const double rho3 = sigma * beta;
        const double rho2 = sigma2 * alpha + gamma * gamma2 * beta;

        beta = v0_.L2Norm();

        const double rho1 = std::sqrt((delta * delta) + (beta * beta));

        for (int i = 0; i < size; ++i)
        {
            w0_[i] = ((1.0 / rho1) * v1_[i]) - ( (rho3 / rho1)  * w0_[i]) - (( rho2 / rho1) * w1_[i]);
        }

        gamma = gamma2;
        gamma2 = delta / rho1;

        for (int i = 0; i < size; ++i)
        {
            x[i] += gamma2 * eta * w0_[i];
        }

        sigma = sigma2;
        sigma2 = beta / rho1;

        eta = -sigma2 * eta;

        if (verbose_)
        {
            printf("MINRES %d: %.2e\n", k, eta);
        }

        if (std::fabs(eta) < tol_)
        {
            break;
        }

        Swap(v0_, v1_);
        Swap(w0_, w1_);
    }
}

Vector<double> MINRES(const Operator& A, const VectorView<double>& b,
                      int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    MINRES(A, b, x, max_iter, tol, verbose);

    return x;
}

void MINRES(const Operator& A, const VectorView<double>& b, VectorView<double>& x,
            int max_iter, double tol, bool verbose)
{
    MINRESSolver minres(A, max_iter, tol, verbose);

    minres.Mult(b, x);
}

PMINRESSolver::PMINRESSolver(const Operator& A, const Operator& M, int max_iter, double tol, bool verbose)
    : A_(A), M_(M), max_iter_(max_iter), tol_(tol), verbose_(verbose),
      w0_(A.Rows()), w1_(A.Rows()),
      v0_(A.Rows()), v1_(A.Rows()),
      u1_(A.Rows()), q_(A.Rows())
{
    assert(A_.Rows() == A_.Cols());
    assert(A_.Rows() == M_.Cols());
    assert(M_.Rows() == M_.Cols());
}

size_t PMINRESSolver::Rows() const
{
    return A_.Rows();
}

size_t PMINRESSolver::Cols() const
{
    return A_.Cols();
}

void PMINRESSolver::Mult(const VectorView<double>& b, VectorView<double>& x) const
{
    assert(b.size() == A_.Rows());
    assert(x.size() == A_.Cols());

    const int size = A_.Cols();

    w0_ = 0.0;
    w1_ = 0.0;
    v0_ = 0.0;

    A_.Mult(x, q_);
    v1_ = b;
    v1_ -= q_;

    M_.Mult(v1_, u1_);

    double beta = u1_.Mult(v1_);
    double eta = beta;

    double gamma = 1.0;
    double gamma2 = 1.0;

    double sigma = 0;
    double sigma2 = 0;

    for (int k = 0; k < max_iter_; ++k)
    {
        v1_ /= beta;
        u1_ /= beta;

        A_.Mult(u1_, q_);

        const double alpha = u1_.Mult(q_);

        for (int i = 0; i < size; ++i)
        {
            v0_[i] = q_[i] - (beta * v0_[i]) - (alpha * v1_[i]);
        }

        const double delta = gamma2 * alpha - gamma * sigma2 * beta;
        const double rho3 = sigma * beta;
        const double rho2 = sigma2 * alpha + gamma * gamma2 * beta;

        M_.Mult(v0_, q_);
        beta = std::sqrt(v0_.Mult(q_));

        const double rho1 = std::sqrt((delta * delta) + (beta * beta));

        for (int i = 0; i < size; ++i)
        {
            w0_[i] = ((1.0 / rho1) * u1_[i]) - ( (rho3 / rho1)  * w0_[i]) - (( rho2 / rho1) * w1_[i]);
        }

        gamma = gamma2;
        gamma2 = delta / rho1;

        for (int i = 0; i < size; ++i)
        {
            x[i] += gamma2 * eta * w0_[i];
        }

        sigma = sigma2;
        sigma2 = beta / rho1;

        eta = -sigma2 * eta;

        if (verbose_)
        {
            printf("PMINRES %d: %.2e\n", k, eta);
        }

        if (std::fabs(eta) < tol_)
        {
            break;
        }

        Swap(u1_, q_);
        Swap(v0_, v1_);
        Swap(w0_, w1_);
    }

}

Vector<double> PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b,
                       int max_iter, double tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    PMINRES(A, M, b, x, max_iter, tol, verbose);

    return x;
}

void PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double>& x,
             int max_iter, double tol, bool verbose)
{
    PMINRESSolver pminres(A, M, max_iter, tol, verbose);

    pminres.Mult(b, x);
}

} //namespace linalgcpp
