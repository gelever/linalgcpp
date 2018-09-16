/*! @file */

#ifndef PARSMOOTHER_HPP__
#define PARSMOOTHER_HPP__

#include "linalgcpp.hpp"
#include "parmatrix.hpp"
#include "parsolvers.hpp"

#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "seq_mv.h"

namespace parlinalgcpp
{

/*  Positive smooth types
    ams.c Explanation
    Relaxation on the ParCSR matrix A with right-hand side f and
    initial guess u. Possible values for relax_type are:

    1 = l1-scaled Jacobi
    2 = l1-scaled block Gauss-Seidel/SSOR
    3 = Kaczmarz

    Negative smooth types
    par_relax.c Explanation
    -----------------------------------------------------------------------
    Switch statement to direct control based on relax_type:
       relax_type = 0 -> Jacobi or CF-Jacobi
       relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
       relax_type = 2 -> Gauss_Seidel: interior points in parallel ,
                      boundary sequential
       relax_type = 3 -> hybrid: SOR-J mix off-processor, SOR on-processor
                with outer relaxation parameters (forward solve)
       relax_type = 4 -> hybrid: SOR-J mix off-processor, SOR on-processor
                with outer relaxation parameters (backward solve)
       relax_type = 5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
       relax_type = 6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
                with outer relaxation parameters
       relax_type = 7 -> Jacobi (uses Matvec), only needed in CGNR
       relax_type = 19-> Direct Solve, (old version)
       relax_type = 29-> Direct solve: use gaussian elimination & BLAS
                (with pivoting) (old version)
    -----------------------------------------------------------------------
*/
enum class SmoothType { L1_Jacobi = 1, L1_GS = 2, Kaczmarz = 3,
                        Jacobi = 0, GS_seq = -1, GS_par = -2,
                        Hybrid_forward = -3, Hybrid_backward = -4,
                        Hybrid_GS = -5, Hybrid_SSOR = -6,
                        Jacobi_matvec = -7, Direct = -19, Direct_blas = -29
                      };


/*! @brief Parallel smoothers from Hypre
    See above enum for smoother types
*/
class ParSmoother: public ParSolver
{
    public:
        /*! @brief Default Empty Constructor */
        ParSmoother();

        /*! @brief Constructor with smoother settings
            @param A Matrix to smooth with
            @param type type of smoothing to apply, see above enum
            @param relax_times number of smoothing steps to apply
            @param relax_weight relaxation weight
            @param omega omega parameter
        */
        ParSmoother(ParMatrix A, SmoothType type = SmoothType::Jacobi,
                    int relax_times = 1, double relax_weight = 1.0,
                    double omega = 1.0);

        /*! @brief Destructor */
        ~ParSmoother();

        /*! @brief Copy Constructor */
        ParSmoother(const ParSmoother& other);

        /*! @brief Move Constructor */
        ParSmoother(ParSmoother&& other);

        /*! @brief Assignment Operator */
        ParSmoother& operator=(ParSmoother other);

        /*! @brief Swap two smoothers
            @param lhs left hand side smoother
            @param rhs right hand side smoother
        */
        friend void swap(ParSmoother& lhs, ParSmoother& rhs) noexcept;

        /*! @brief Apply the smoother Mx = y
            @param input input vector x
            @param output output vector y
        */
        void Mult(const linalgcpp::VectorView<double>& input,
                  linalgcpp::VectorView<double> output) const;
    private:
        SmoothType type_;
        int relax_times_;
        double relax_weight_;
        double omega_;

        mutable linalgcpp::Vector<double> buffer_;
        mutable hypre_ParVector* buffer_shell_;
        mutable std::vector<HYPRE_Real> l1_norms_;
};

} //namespace parlinalgcpp

#endif // PARSMOOTHER_HPP
