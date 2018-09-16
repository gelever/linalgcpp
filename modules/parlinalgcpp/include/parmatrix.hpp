/*! @file ParMatrix header */

#ifndef PARMATRIX_HPP__
#define PARMATRIX_HPP__

#include "mpi.h"
#include "linalgcpp.hpp"
#include "parvector.hpp"
#include "parutilities.hpp"
#include "paroperator.hpp"
#include "parcommpkg.hpp"

#include "_hypre_parcsr_mv.h"
#include "seq_mv.h"

namespace parlinalgcpp
{

class ParVector;

/*! @brief Distributed matrix using Hypre's format

    Each processor owns a continuous, non-empty, subset
    of the global rows and columns. This is tracked by the respective
    row or column starts array.

    Each row contains two non-intersecting blocks.
    Local entries whose column index is owned by the processor are placed
    in the diagonal block. All other entries are considered off diagonal.
*/

class ParMatrix: public ParOperator
{
    public:
        /*! @brief Default Constructor */
        ParMatrix();

        /*! @brief Block Diagonal Constructor
            @param comm Communicator for this matrix
            @param diag Diagonal block

            @note Offsets are computed, requires communication.
        */
        ParMatrix(MPI_Comm comm,
                  linalgcpp::SparseMatrix<double> diag);

        /*! @brief General Constructor
            @param comm Communicator for this matrix
            @param diag Diagonal block
            @param offd Off-diagonal block
            @param col_map Global indices of offd block,
                   these must be sorted

            @note Offsets are computed, requires communication.
        */
        ParMatrix(MPI_Comm comm,
                  linalgcpp::SparseMatrix<double> diag,
                  linalgcpp::SparseMatrix<double> offd,
                  std::vector<HYPRE_Int> col_map);

        /*! @brief Square Block Diagonal Constructor
            @param comm Communicator for this matrix
            @param starts Both row and column processor starts
            @param diag Diagonal block
        */
        ParMatrix(MPI_Comm comm,
                  std::vector<HYPRE_Int> starts,
                  linalgcpp::SparseMatrix<double> diag);

        /*! @brief Rectangle Block Diagonal Constructor
            @param comm Communicator for this matrix
            @param row_starts processor row partition scheme
            @param col_starts processor column partition scheme
            @param diag diagonal block
        */
        ParMatrix(MPI_Comm comm,
                  std::vector<HYPRE_Int> row_starts,
                  std::vector<HYPRE_Int> col_starts,
                  linalgcpp::SparseMatrix<double> diag);

        /*! @brief General Constructor
            @param comm Communicator for this matrix
            @param row_starts processor row partition scheme
            @param col_starts processor column partition scheme
            @param diag diagonal block
            @param offd Off-diagonal block
            @param col_map Global indices of offd block,
                   these must be sorted
        */
        ParMatrix(MPI_Comm comm,
                  std::vector<HYPRE_Int> row_starts,
                  std::vector<HYPRE_Int> col_starts,
                  linalgcpp::SparseMatrix<double> diag,
                  linalgcpp::SparseMatrix<double> offd,
                  std::vector<HYPRE_Int> col_map);

        /*! @brief Copy Constructor */
        ParMatrix(const ParMatrix& other) noexcept;

        /*! @brief Move Constructor */
        ParMatrix(ParMatrix&& other) noexcept;

        /*! @brief Assignment Constructor */
        ParMatrix& operator=(ParMatrix other) noexcept;

        /*! @brief Default Destructor */
        ~ParMatrix() noexcept;

        /*! @brief Swap two matrices
            @param lhs Left hand side matrix
            @param rhs Right hand side matrix
        */
        friend void swap(ParMatrix& lhs, ParMatrix& rhs) noexcept;

        /*! @brief Compute transpose of this matrix */
        ParMatrix Transpose() const;

        /*! @brief Multiply matrix on the right
            @param other ParMatrix to multiply
        */
        ParMatrix Mult(const ParMatrix& other) const;

        /*! @brief Multiply matrix on the right
            @param other ParMatrix to multiply
        */
        ParMatrix operator*(const ParMatrix& other) const;

        /*! @brief Scale all non-zero entries by given value
            @param val Scale to apply
        */
        ParMatrix& operator*=(double val);

        /*! @brief Set all non-zero entries to given value
            @param val Value to set
        */
        ParMatrix& operator=(double val);

        /*! @brief Computes negative of this matrix */
        ParMatrix operator-() const;

        /*! @brief Multiply a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        void Mult(const linalgcpp::VectorView<double>& input,
                  linalgcpp::VectorView<double> output) const;

        /*! @brief Multiply a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        linalgcpp::Vector<double> Mult(const linalgcpp::VectorView<double>& input) const;

        /*! @brief Multiply a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        void Mult(const ParVector& input, ParVector& output) const;

        /*! @brief Multiply a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        ParVector Mult(const ParVector& input) const;

        /*! @brief Multiply a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        void MultAT(const linalgcpp::VectorView<double>& input,
                    linalgcpp::VectorView<double> output) const;

        /*! @brief Multiply a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        linalgcpp::Vector<double> MultAT(const linalgcpp::VectorView<double>& input) const;

        /*! @brief Multiply a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        void MultAT(const ParVector& input, ParVector& output) const;

        /*! @brief Multiply a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        ParVector MultAT(const ParVector& input) const;

        /*! @brief Access the diagonal block */
        const linalgcpp::SparseMatrix<double>& GetDiag() const;

        /*! @brief Access the off-diagonal block */
        const linalgcpp::SparseMatrix<double>& GetOffd() const;

        /*! @brief Access the column map */
        const std::vector<HYPRE_Int>& GetColMap() const;

        /*! @brief Print entries in both blocks and the column map
            @param label Label to show prior to printing
            @param out Output stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Cast to Hypre's format */
        operator hypre_ParCSRMatrix* ()
        {
            return A_;
        }

        /*! @brief Cast to Hypre's format */
        operator const hypre_ParCSRMatrix* () const
        {
            return A_;
        }

        /*! @brief Add a scalar value to the diagonal of the diagonal block
            @param diag_val Value to add

            @note diagonal entries must already exist
        */
        void AddDiag(double diag_val);

        /*! @brief Add given values to the diagonal of the diagonal block
            @param diag_vals Values to add

            @note diagonal entries must already exist
        */
        void AddDiag(const std::vector<double>& diag_vals);

        /*! @brief Create copy of Communication Package */
        ParCommPkg MakeCommPkg() const;

        /*! @brief Compute max norm: the maximum of the absolute values
            of all entries.
            @retval global_max Global max norm
        */
        double MaxNorm() const;

        /*! @brief Global number of non-zero entries
            @retval global_nnz Global number of non-zero entries
        */
        int nnz() const;

        /*! @brief Scale rows by diagonal matrix
            @param values scale per row
        */
        void ScaleRows(const linalgcpp::SparseMatrix<double>& values);

        /*! @brief Scale rows by inverse of diagonal matrix
            @param values scale per row
        */
        void InverseScaleRows(const linalgcpp::SparseMatrix<double>& values);

        /*! @brief Scale rows by given values
            @param values scale per row
        */
        void ScaleRows(const std::vector<double>& values);

        /*! @brief Scale rows by inverse of given values
            @param values scale per row
        */
        void InverseScaleRows(const std::vector<double>& values);

        /*! @brief Eliminate a row by setting all row entries to zero
            @param index row to eliminate
        */
        void EliminateRow(int index);

    private:
        void Init();
        void HypreCreate();
        void HypreDestroy();

        hypre_ParCSRMatrix* A_;

        linalgcpp::SparseMatrix<double> diag_;
        linalgcpp::SparseMatrix<double> offd_;

        std::vector<HYPRE_Int> col_map_;
};

inline
const std::vector<HYPRE_Int>& ParMatrix::GetColMap() const
{
    return col_map_;
}

inline
const linalgcpp::SparseMatrix<double>& ParMatrix::GetDiag() const
{
    return diag_;
}

inline
const linalgcpp::SparseMatrix<double>& ParMatrix::GetOffd() const
{
    return offd_;
}

inline
ParMatrix ParMatrix::operator*(const ParMatrix& other) const
{
    return Mult(other);
}

} //namespace parlinalgcpp

#endif // PARMATRIX_HPP
