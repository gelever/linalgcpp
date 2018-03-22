/*! @file */

#ifndef SOLVERS_HPP__
#define SOLVERS_HPP__

#include "operator.hpp"
#include "vector.hpp"

namespace linalgcpp
{

/*! @class Conjugate Gradient.  Solves Ax = b for positive definite A */
class CGSolver : public Operator
{
    public:
        /*! @brief Constructor
            @param A operator to apply the action of A
            @param max_iter maxiumum number of iterations to perform
            @param tol relative tolerance for stopping criteria
            @param verbose display additional iteration information
        */
        //CGSolver(const Operator& A, int max_iter = 1000, double tol = 1e-16, bool verbose = false);
        CGSolver(const Operator& A, int max_iter = 1000, double tol = 1e-16, bool verbose = false,
                 double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        const Operator& A_;
        int max_iter_;
        double tol_;
        bool verbose_;

        mutable Vector<double> Ap_;
        mutable Vector<double> r_;
        mutable Vector<double> p_;

        double (*Dot_)(const VectorView<double>&, const VectorView<double>&);
};

class PCGSolver : public Operator
{
    public:
        /*! @brief Constructor
            @param A operator to apply the action of A
            @param M operator to apply the preconditioner
            @param max_iter maxiumum number of iterations to perform
            @param tol relative tolerance for stopping criteria
            @param verbose display additional iteration information
        */
        PCGSolver(const Operator& A, const Operator& M, int max_iter = 1000, double tol = 1e-16, bool verbose = false);

        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        const Operator& A_;
        const Operator& M_;
        int max_iter_;
        double tol_;
        bool verbose_;

        mutable Vector<double> Ap_;
        mutable Vector<double> r_;
        mutable Vector<double> p_;
        mutable Vector<double> z_;
};

class MINRESSolver : public Operator
{
    public:
        /*! @brief Constructor
            @param A operator to apply the action of A
            @param max_iter maxiumum number of iterations to perform
            @param tol relative tolerance for stopping criteria
            @param verbose display additional iteration information
        */
        MINRESSolver(const Operator& A, int max_iter = 1000, double tol = 1e-16, bool verbose = false,
                 double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        const Operator& A_;

        int max_iter_;
        double tol_;
        bool verbose_;

        mutable Vector<double> w0_;
        mutable Vector<double> w1_;
        mutable Vector<double> v0_;
        mutable Vector<double> v1_;
        mutable Vector<double> q_;

        double (*Dot_)(const VectorView<double>&, const VectorView<double>&);
};

class PMINRESSolver : public Operator
{
    public:
        /*! @brief Constructor
            @param A operator to apply the action of A
            @param M operator to apply the preconditioner
            @param max_iter maxiumum number of iterations to perform
            @param tol relative tolerance for stopping criteria
            @param verbose display additional iteration information
        */
        PMINRESSolver(const Operator& A, const Operator& M, int max_iter = 1000, double tol = 1e-16, bool verbose = false,
                 double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        const Operator& A_;
        const Operator& M_;

        int max_iter_;
        double tol_;
        bool verbose_;

        mutable Vector<double> w0_;
        mutable Vector<double> w1_;
        mutable Vector<double> v0_;
        mutable Vector<double> v1_;
        mutable Vector<double> u1_;
        mutable Vector<double> q_;

        double (*Dot_)(const VectorView<double>&, const VectorView<double>&);
};


/*! @brief Conjugate Gradient.  Solves Ax = b
    @param A operator to apply the action of A
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    @note Uses random initial guess for x
*/
Vector<double> CG(const Operator& A, const VectorView<double>& b,
                  int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Conjugate Gradient.  Solves Ax = b for positive definite A
    @param A operator to apply the action of A
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
*/
void CG(const Operator& A, const VectorView<double>& b, VectorView<double> x,
        int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Preconditioned Conjugate Gradient.  Solves Ax = b
           where M is preconditioner for A
    @param A operator to apply the action of A
    @param M operator to apply the preconditioner
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    @note Uses random initial guess for x
*/
Vector<double> PCG(const Operator& A, const Operator& M, const VectorView<double>& b,
                   int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Preconditioned Conjugate Gradient.  Solves Ax = b
           where M is preconditioner for A
    @param A operator to apply the action of A
    @param M operator to apply the preconditioner
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
*/
void PCG(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double> x,
         int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    Modified from mfem implementation
    @note Uses random initial guess for x
*/
Vector<double> MINRES(const Operator& A, const VectorView<double>& b,
                      int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information

    Modified from mfem implementation
*/

void MINRES(const Operator& A, const VectorView<double>& b, VectorView<double> x,
            int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Preconditioned MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param M operator to apply of the preconditioner
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    Modified from mfem implementation
    @note Uses random initial guess for x
*/
Vector<double> PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b,
                       int max_iter = 1000, double tol = 1e-16, bool verbose = false);

/*! @brief Preconditioned MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param M operator to apply of the preconditioner
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param tol relative tolerance for stopping criteria
    @param verbose display additional iteration information

    Modified from mfem implementation
*/
void PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double> x,
             int max_iter = 1000, double tol = 1e-16, bool verbose = false);

} //namespace linalgcpp

#endif // SOLVERS_HPP__
