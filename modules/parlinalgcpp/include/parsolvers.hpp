/*! @file */

#ifndef PARSOLVERS_HPP__
#define PARSOLVERS_HPP__

#include "linalgcpp.hpp"
#include "parmatrix.hpp"
#include "paroperator.hpp"
#include "parvector.hpp"
#include "hypre_ops.hpp"

#include "temp_multivector.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "seq_mv.h"

namespace linalgcpp
{

inline HYPRE_Int no_create(HYPRE_Solver*)
{
    return 0;
};
inline HYPRE_Int no_destroy(HYPRE_Solver)
{
    return 0;
};
inline HYPRE_Int no_setup(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector)
{
    return 0;
};
inline HYPRE_Int no_solve(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector)
{
    return 0;
};


/*! @brief Base class for parallel solvers
    Solves system Ax = y
*/
class ParSolver: public ParOperator
{
    public:
        /*! @brief Default Constructor */
        ParSolver();

        /*! @brief Constructor with solver settings
            @param A Matrix A in Ax = y
            @param max_iter maximum iterations to perform
            @param tol solve tolerance
            @param verbose show verbose output
            @param create Hypre create function for the solver
            @param destroy Hypre destroy function for the solver
            @param setup Hypre setup function for the solver
            @param solve Hypre solve function for the solver
        */
        ParSolver(ParMatrix A,
                  size_t max_iter = 0, double tol = 0.0, bool verbose = false,
                  HYPRE_Int (*create)(HYPRE_Solver*) = &no_create,
                  HYPRE_Int (*destroy)(HYPRE_Solver) = &no_destroy,
                  HYPRE_Int (*setup)(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                     HYPRE_ParVector, HYPRE_ParVector) = &no_setup,
                  HYPRE_Int (*solve)(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                     HYPRE_ParVector, HYPRE_ParVector) = &no_solve);

        /*! @brief Copy Constructor */
        ParSolver(const ParSolver& other);

        /*! @brief Move Constructor */
        ParSolver(ParSolver&& other);

        /*! @brief Assignment Constructor */
        virtual ParSolver& operator=(ParSolver A);

        /*! @brief Swap two solvers */
        friend void swap(ParSolver& lhs, ParSolver& rhs) noexcept;

        /*! @brief Destructor */
        virtual ~ParSolver();

        /*! @brief Apply the action of the solver
            @param input input vector x
            @param output output vector y
        */
        virtual void Mult(const linalgcpp::VectorView<double>& input,
                          linalgcpp::VectorView<double> output) const;

        /*! @brief Access the hypre solver */
        HYPRE_Solver& GetSolver()
        {
            return solver_;
        }

        /*! @brief Access the hypre solver */
        const HYPRE_Solver& GetSolver() const
        {
            return solver_;
        }

        /*! @brief Access the hypre solver */
        operator HYPRE_Solver() const
        {
            return solver_;
        }

        using hypre_sig = HYPRE_Int (*)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
        /*! @brief Access the hypre solve function */
        hypre_sig GetSolveFn() const
        {
            return solve_;
        }

        /*! @brief Access the hypre setup function */
        hypre_sig GetSetupFn() const
        {
            return setup_;
        }

        /*! @brief Access the matrix associated w/ this solver */
        const ParMatrix& GetMatrix() const
        {
            return A_;
        }

    protected:
        void HypreSetup();
        size_t max_iter_;
        double tol_;
        bool verbose_;

        ParMatrix A_;
    private:
        void HypreCreate();

        HYPRE_Int (*create_)(HYPRE_Solver*);
        HYPRE_Int (*destroy_)(HYPRE_Solver);
        HYPRE_Int (*setup_)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
        HYPRE_Int (*solve_)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);

        HYPRE_Solver solver_;
};

class BoomerAMG: public ParSolver
{
    public:
        BoomerAMG() = default;
        BoomerAMG(ParMatrix A,
                  size_t max_iter = 1,
                  double tol = 0.0, bool verbose = false);

        BoomerAMG(const BoomerAMG& other) noexcept;
        BoomerAMG(BoomerAMG&& other) noexcept;
        BoomerAMG& operator=(BoomerAMG other) noexcept;

        friend void swap(BoomerAMG& lhs, BoomerAMG& rhs) noexcept;

        ~BoomerAMG() = default;
};

class ParaSails: public ParSolver
{
    public:
        ParaSails() = default;
        ParaSails(ParMatrix A, bool verbose = false,
                  int reuse = 0, int sym = 0,
                  double load_bal = 0.0,
                  double threshold = 0.1,
                  double filter = 0.1,
                  int max_levels = 1);

        ParaSails(const ParaSails& other) noexcept;
        ParaSails(ParaSails&& other) noexcept;
        ParaSails& operator=(ParaSails other) noexcept;

        friend void swap(ParaSails& lhs, ParaSails& rhs) noexcept;

        ~ParaSails() = default;

        void SetFilter(double filter);
        void SetSym(int sym);
        void SetLoadBal(int load_bal);
        void SetReuse(int reuse);
        void SetLogging(int logging);
};

class ParCG: public ParSolver
{
    public:
        ParCG() = default;
        ParCG(ParMatrix A, size_t max_iter = 1000,
              double tol = 1e-8, bool verbose = false);
        ParCG(ParMatrix A, const ParSolver& M, size_t max_iter = 1000,
              double tol = 1e-8, bool verbose = false);

        ParCG(const ParCG& other) noexcept;
        ParCG(ParCG&& other) noexcept;
        ParCG& operator=(ParCG other) noexcept;

        friend void swap(ParCG& lhs, ParCG& rhs) noexcept;

        ~ParCG() = default;

        void EnableDiagScaling();
};

class ParDiagScale: public ParSolver
{
    public:
        ParDiagScale() = default;
        ParDiagScale(ParMatrix A, size_t max_iter = 1000,
                     double tol = 1e-8, bool verbose = false);

        ParDiagScale(const ParDiagScale& other) noexcept;
        ParDiagScale(ParDiagScale&& other) noexcept;
        ParDiagScale& operator=(ParDiagScale other) noexcept;

        friend void swap(ParDiagScale& lhs, ParDiagScale& rhs) noexcept;

        ~ParDiagScale() = default;
};

std::vector<double> LOBPCG(const ParOperator& pmat, std::vector<linalgcpp::Vector<double>>& x,
                           const linalgcpp::Operator* precond = nullptr, bool verbose = false);
std::vector<double> LOBPCG(const ParOperator& pmat, int num_evects = 1, const linalgcpp::Operator* precond = nullptr,
                           bool verbose = false);

} //namespace linalgcpp

#endif // PARMATRIX_HPP
