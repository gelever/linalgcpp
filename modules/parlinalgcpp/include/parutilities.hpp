/*! @file */

#ifndef PARUTILITIES_HPP__
#define PARUTILITIES_HPP__

#include "seq_mv.h"

#include "coomatrix.hpp"
#include "parmatrix.hpp"
#include "parvector.hpp"

namespace linalgcpp
{

/// Call output only on processor 0
#define ParPrint(myid, output) if (myid == 0) { output; }

/*! @brief Throw if false in debug mode only */
template <typename F>
void linalgcpp_parassert(MPI_Comm comm, F&& lambda, const std::string& message = "linalgcpp assertion failed")
{
#ifndef NDEBUG
    int local_exp = !lambda();
    int global_exp;
    MPI_Allreduce(&local_exp, &global_exp, 1, MPI_INT, MPI_SUM, comm);

    if (global_exp)
    {
        throw std::runtime_error(message);
    }
#endif // NDEBUG
}

/*! @brief Throw if false unconditionally */
template <typename F>
void linalgcpp_parverify(MPI_Comm comm, F&& lambda, const std::string& message = "linalgcpp verification failed")
{
    int local_exp = !lambda();
    int global_exp;
    MPI_Allreduce(&local_exp, &global_exp, 1, MPI_INT, MPI_SUM, comm);

    if (global_exp)
    {
        throw std::runtime_error(message);
    }
}

/*! @brief Throw if false unconditionally */
inline
void linalgcpp_parverify(MPI_Comm comm, bool expression, const std::string& message = "linalgcpp verification failed")
{
    int local_exp = !expression;
    int global_exp;
    MPI_Allreduce(&local_exp, &global_exp, 1, MPI_INT, MPI_SUM, comm);

    if (global_exp)
    {
        throw std::runtime_error(message);
    }
}



class ParMatrix;

/// Split a square matrix between processes
ParMatrix ParSplit(MPI_Comm comm, const linalgcpp::SparseMatrix<double>& A_global, const std::vector<int>& proc_part);

/// Split a square matrix between processes, including local to global map
ParMatrix ParSplit(MPI_Comm comm, const linalgcpp::SparseMatrix<double>& A_global, const std::vector<int>& proc_part, std::vector<int>& local_part);

/// Generate offsets given local sizes
std::vector<HYPRE_Int> GenerateOffsets(MPI_Comm comm, int local_size);

/// Generate multiple offsets given local sizes
std::vector<std::vector<HYPRE_Int>> GenerateOffsets(MPI_Comm comm, const std::vector<int>& local_sizes);

/// Sort a column map and permute the corresponding coo matrix
void SortColumnMap(std::vector<HYPRE_Int>& col_map, linalgcpp::CooMatrix<double>& coo_offd);

/// Compute C = A * B
ParMatrix Mult(const ParMatrix& lhs, const ParMatrix& rhs);

/// Compute C = A^T
ParMatrix Transpose(const ParMatrix& mat);

/// Compute C = R^T * A * P
ParMatrix RAP(const ParMatrix& R, const ParMatrix& A, const ParMatrix& P);

/// Compute C = P^T * A * P
ParMatrix RAP(const ParMatrix& A, const ParMatrix& P);

/// Compute C = A + B
/** @note Row starts must match between A and B */
ParMatrix ParAdd(const ParMatrix& A, const ParMatrix& B);

/// Compute C = (alpha * A) + (beta * B)
/** @note Row starts must match between A and B */
ParMatrix ParAdd(double alpha, const ParMatrix& A, double beta, const ParMatrix& B);

/// Compute C = (alpha * A) + (beta * B)
/** @note Row starts must match between A and B */
/** @note Uses array of size global columns as workspace,
    reuse this array if multiple additions. */
/** @note This may be made more efficient, goes through
    CooMatrix instead of building in place */
ParMatrix ParAdd(double alpha, const ParMatrix& A, double beta, const ParMatrix& B,
                 std::vector<int>& marker);

/// Compute C = A - B
/** @note Row starts must match between A and B */
ParMatrix ParSub(const ParMatrix& A, const ParMatrix& B);

/// Compute C = (alpha * A) - (beta * B)
/** @note Row starts must match between A and B */
/** @note Uses array of size global columns as workspace,
    reuse this array if multiple additions. */
ParMatrix ParSub(double alpha, const ParMatrix& A, double beta, const ParMatrix& B);

/// Compute C = (alpha * A) - (beta * B)
/** @note Row starts must match between A and B */
ParMatrix ParSub(double alpha, const ParMatrix& A, double beta, const ParMatrix& B, std::vector<int>& marker);

/// Othogonalize against constant vector
void ParSubAvg(MPI_Comm comm, VectorView<double> x);

/// Othogonalize against constant vector, with known global size
void ParSubAvg(MPI_Comm comm, VectorView<double> x, int global_size);

/** @brief Handles mpi initialization and finalization */
struct MpiSession
{
    /** @brief Constructor

        @param argc argc from command line
        @param argv argv from command line
        @param comm MPI Communicator to use
    */
    MpiSession(int argc, char** argv, MPI_Comm comm_in = MPI_COMM_WORLD)
        : comm(comm_in)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(comm, &num_procs);
        MPI_Comm_rank(comm, &myid);
    }

    /// Do not allow initializing mpi multiple times
    MpiSession(const MpiSession& other) = delete;
    MpiSession(MpiSession&& other) = delete;
    MpiSession& operator=(const MpiSession& other) = delete;

    /** @brief Destructor */
    ~MpiSession() { MPI_Finalize(); }

    MPI_Comm comm;
    int num_procs;
    int myid;
};

} // namespace linalgcpp
#endif // PARUTILITIES_HPP__
