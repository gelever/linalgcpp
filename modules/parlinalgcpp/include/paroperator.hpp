/*! @file */

#ifndef PAROPERATOR_HPP__
#define PAROPERATOR_HPP__

#include "linalgcpp.hpp"

#include "temp_multivector.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "seq_mv.h"

namespace linalgcpp
{


/*! @brief Base class for distributed operators.

    Each processor owns a continuous, non-empty, subset
    of the global rows and columns. This is tracked by the respective
    row or column starts array.
*/

class ParOperator: public linalgcpp::Operator
{
    public:
        /*! @brief Default Constructor */
        ParOperator();

        /*! @brief Comm Constructor
            @param comm Communicator for this operator
            @note both row and column processor partitionings must be set by
                  derived class!
        */
        ParOperator(MPI_Comm);

        /*! @brief Square Constructor
            @param comm Communicator for this operator
            @param starts both row and column processor partitioning
        */
        ParOperator(MPI_Comm, std::vector<HYPRE_Int> starts);

        /*! @brief Rectangular Constructor
            @param comm Communicator for this operator
            @param row_starts row processor partitioning
            @param col_starts column processor partitioning
        */
        ParOperator(MPI_Comm, std::vector<HYPRE_Int> row_starts, std::vector<HYPRE_Int> col_starts);

        /*! @brief Copy Constructor */
        ParOperator(const ParOperator& other);

        /*! @brief Destructor */
        virtual ~ParOperator();

        /*! @brief Swap two operators
            @param lhs left hand side operator
            @param rhs right hand side operator
        */
        friend void swap(ParOperator& lhs, ParOperator& rhs) noexcept;

        /*! @brief Apply the action of this operator Ax = y
            @param input vector x
            @param output vector y
        */
        virtual void Mult(const linalgcpp::VectorView<double>& input,
                          linalgcpp::VectorView<double> output) const = 0;

        /*! @brief Apply the action of this operator Ax = y
            @param input vector x
            @param output vector y
        */
        virtual Vector<double> Mult(const linalgcpp::VectorView<double>& input) const;

        /* @brief Global number of rows */
        virtual int GlobalRows() const;

        /* @brief Global number of cols */
        virtual int GlobalCols() const;

        /* @brief Access the row processor partioning */
        virtual const std::vector<HYPRE_Int>& GetRowStarts() const;

        /* @brief Access the column processor partioning */
        virtual const std::vector<HYPRE_Int>& GetColStarts() const;

        /* @brief Access the MPI communicator */
        virtual MPI_Comm GetComm() const;

        /* @brief Access the MPI processor id */
        virtual int GetMyId() const;

        /* @brief Access the number of MPI processors */
        virtual int GetNumProcs() const;

    protected:
        MPI_Comm comm_;
        int myid_;
        int num_procs_;

        std::vector<HYPRE_Int> row_starts_;
        std::vector<HYPRE_Int> col_starts_;
        hypre_ParVector* x_;
        hypre_ParVector* b_;

        void InitVector(hypre_ParVector*& vect, std::vector<HYPRE_Int>& starts);
        void DestroyVector(hypre_ParVector*& vect);

    private:
        void HypreCreate();
        void HypreDestroy();
};

} //namespace linalgcpp

#endif // PAROPERATOR_HPP
