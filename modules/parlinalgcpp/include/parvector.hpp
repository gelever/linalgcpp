/*! @file */

#ifndef PARVECTOR_HPP__
#define PARVECTOR_HPP__

#include "mpi.h"
#include "linalgcpp.hpp"
#include "parmatrix.hpp"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"

namespace linalgcpp
{

/*! @brief Distributed vector using Hypre's format

    Each processor owns a continuous, non-empty, subset
    of the global vector.
*/
class ParVector: public linalgcpp::Vector<double>
{
        friend class ParMatrix;
    public:
        /*! @brief Default Constructor */
        ParVector();

        /*! @brief Constructor setting only the vector partition
            @param comm MPI Communicator
            @param part processor partition
        */
        ParVector(MPI_Comm comm, const std::vector<HYPRE_Int>& part);

        /*! @brief Constructor setting only the vector partition and initial vector value
            @param comm MPI Communicator
            @param val initial vector value
            @param part processor partition
        */
        ParVector(MPI_Comm comm, double val, const std::vector<HYPRE_Int>& part);

        /*! @brief Constructor copying a local vector
            @param comm MPI Communicator
            @param vect local vector to copy
            @param part processor partition
        */
        ParVector(MPI_Comm comm, const linalgcpp::VectorView<double>& vect, const std::vector<HYPRE_Int>& part);

        /*! @brief Copy Constructor */
        ParVector(const ParVector& other);

        /*! @brief Move Constructor */
        ParVector(ParVector&& other);

        /*! @brief Assignment Constructor */
        ParVector& operator=(ParVector other);

        /*! @brief Destructor */
        ~ParVector() noexcept;

        /*! @brief Swap two vectors */
        friend void swap(ParVector& lhs, ParVector& rhs) noexcept;

        /*! @brief Get global size of vector */
        int GlobalSize() const;

        /*! @brief Set the vector to a constant value
            @param val constant value to set
        */
        ParVector& operator=(double val);

        /*! @brief Compute global vector inner product
            @param vect other vector in inner product
            @returns double result of inner product
        */
        virtual double Mult(const VectorView<double>& vect) const override;

        /*! @brief Compute global l2 norm */
        double L2Norm() const override;

        /*! @brief Compute local sum of vector entries */
        double LocalSum() const;

        /*! @brief Compute global sum of vector entries */
        double GlobalSum() const;

    private:
        void Init();

        void HypreCreate();
        void HypreDestroy();

        MPI_Comm comm_;
        hypre_ParVector* pvect_;
        std::vector<HYPRE_Int> partition_;

        int global_size_;
};

// Utility functions


/*! @brief Add right hand side vector to left hand side
    @param lhs left hand side
    @param rhs right hand side
    @returns left hand side plus right hand side
*/
ParVector& operator+=(ParVector& lhs, const ParVector& rhs);

/*! @brief Subtract right hand side vector from left hand side
    @param lhs left hand side
    @param rhs right hand side
    @returns left hand side minus right hand side
*/
ParVector& operator-=(ParVector& lhs, const ParVector& rhs);

/*! @brief Entry multiply right hand side vector with left hand side
    @param lhs left hand side
    @param rhs right hand side
    @returns left hand side multiplied entrywise with right hand side
*/
ParVector& operator*=(ParVector& lhs, const ParVector& rhs);

/*! @brief Entry divide left hand side vector by right hand side
    @param lhs left hand side
    @param rhs right hand side
    @returns left hand side divided entrywise by right hand side
*/
ParVector& operator/=(ParVector& lhs, const ParVector& rhs);

/*! @brief Global vector inner product
    @param lhs left hand side vector
    @param rhs right hand side vector
    @returns global inner product
*/
double Mult(const ParVector& lhs, const ParVector& rhs);

/*! @brief Compute the global l2 norm of local vectors
    @param comm MPI Communicator
    @param vector local vector
    @returns global l2 norm
*/
double ParL2Norm(MPI_Comm comm, const linalgcpp::VectorView<double>& vect);

/*! @brief Compute the global l2 norm
    @param vect global vector
    @returns global l2 norm
*/
double L2Norm(const ParVector& vect);

/*! @brief Subtract the average constant vector
    @param vect global vector
*/
void SubAvg(ParVector& vect);

/*! @brief Compute the global vector inner product using local vectors
    @param comm MPI Communicator
    @param lhs local left hand side vector
    @param rhs local right hand side vector
    @returns global vector inner product
*/
double ParMult(MPI_Comm comm,
               const linalgcpp::VectorView<double>& lhs,
               const linalgcpp::VectorView<double>& rhs);

/*! @brief Compute the global vector inner product using local vectors
    @param lhs local left hand side vector
    @param rhs local right hand side vector
    @returns global vector inner product
*/
inline
double ParMult(const linalgcpp::VectorView<double>& lhs,
               const linalgcpp::VectorView<double>& rhs)
{
    return ParMult(MPI_COMM_WORLD, lhs, rhs);
}

} // namespace linalgcpp

#endif

