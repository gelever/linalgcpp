/*! @file */

#ifndef EIGENSOLVER_HPP__
#define EIGENSOLVER_HPP__

#include <tuple>

#include "densematrix.hpp"
#include "sparsematrix.hpp"

namespace linalgcpp
{

using EigenPair = std::pair<std::vector<double>, DenseMatrix>;

/*! @brief Dense matrix eigensolver using lapack
           Calculate smallest eigenvalues and eigenvectors corresponding
           to given matrix. Keeps eigenpiars up to given max eigenvectors
           or relative tolerance.
    
    Modified from the smoothG implementation at
    https://github.com/LLNL/smoothG/blob/master/src/utilities.cpp
*/
class EigenSolver
{

    public:
        /*! @brief Default Constructor*/
        EigenSolver();

        /*! @brief Copy Constructor */
        EigenSolver(const EigenSolver& other) = default;

        /*! @brief Move Constructor */
        EigenSolver(EigenSolver&& other) = default;

        /*! @brief Assignment Operator */
        EigenSolver& operator=(const EigenSolver& other) = default;

        /*! @brief Default Destructor*/
        ~EigenSolver() = default;

        /*! @brief Compute select number of lower spectrum eigen pairs

            @param mat Converts sparse matrix to dense matrix
            @param rel_tol Relative tolerance to max eigenvalue
            @param max_evect Maximum number of eigenvectors to keep
            @returns EigenPair pair of eigenvalues and eigenvectors
        */
        EigenPair Solve(const SparseMatrix<double>& mat, double rel_tol, int max_evect);

        /*! @brief Compute select number of lower spectrum eigen pairs

            @param mat Dense matrix to solve for
            @param rel_tol Relative tolerance to max eigenvalue
            @param max_evect Maximum number of eigenvectors to keep
            @returns EigenPair pair of eigenvalues and eigenvectors
        */
        EigenPair Solve(const DenseMatrix& mat, double rel_tol, int max_evect);

        /*! @brief Compute select number of lower spectrum eigen pairs

            @param mat Converts sparse matrix to dense matrix
            @param rel_tol Relative tolerance to max eigenvalue
            @param max_evect Maximum number of eigenvectors to keep
            @param EigenPair pair of eigenvalues and eigenvectors
        */
        void Solve(const SparseMatrix<double>& mat, double rel_tol, int max_evect,
                   EigenPair& eigen_pair);

        /*! @brief Compute select number of lower spectrum eigen pairs

            @param mat Dense matrix to solve for
            @param rel_tol Relative tolerance to max eigenvalue
            @param max_evect Maximum number of eigenvectors to keep
            @param EigenPair pair of eigenvalues and eigenvectors
        */
        void Solve(const DenseMatrix& mat, double rel_tol, int max_evect,
                   EigenPair& eigen_pair);
                  

    private:
        void AllocateWorkspace(int size);

        char uplo_;
        char side_;
        char trans_;
        double abstol_;

        int info_;
        int alloc_size_;
        int lwork_;

        std::vector<double> A_;
        std::vector<double> work_;
        std::vector<int> iwork_;

        // Triangularization info
        std::vector<double> d_;
        std::vector<double> e_;
        std::vector<double> tau_;

        // Block info for dstein_ / dormtr_
        std::vector<int> iblock_;
        std::vector<int> isplit_;
        std::vector<int> ifail_;
};

} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
