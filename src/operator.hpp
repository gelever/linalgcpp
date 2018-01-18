/*! @file */

#ifndef OPERATOR_HPP__
#define OPERATOR_HPP__

#include <memory>
#include <vector>
#include <numeric>
#include <assert.h>

#include "vector.hpp"

namespace linalgcpp
{

/*! @brief Abstract operator class that has a size
          and can apply its action to a vector
*/
class Operator
{
    public:
        /*! @brief Default Constructor */
        Operator() = default;

        /*! @brief Destructor */
        virtual ~Operator() noexcept = default;

        /*! @brief The number of rows in this operator
            @retval the number of rows
        */
        virtual size_t Rows() const = 0;

        /*! @brief The number of columns in this operator
            @retval the number of columns
        */
        virtual size_t Cols() const = 0;

        /*! @brief Apply the action to a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        virtual void Mult(const VectorView<double>& input, VectorView<double>& output) const = 0;

        /*! @brief Apply the action to a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        virtual Vector<double> Mult(const VectorView<double>& input) const;

        /*! @brief Apply the transpose action to a vector: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        virtual void MultAT(const VectorView<double>& input, VectorView<double>& output) const;

        /*! @brief Apply the transpose action to a vector: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        virtual Vector<double> MultAT(const VectorView<double>& input) const;

        /*! @brief Computes the inner produt y^T A x
            @param x the input vector x
            @param y the input vector y
            @retval double the A inner product
        */
        double InnerProduct(const VectorView<double>& x, const VectorView<double>& y) const;
};

inline
void Operator::MultAT(const VectorView<double>& input, VectorView<double>& output) const
{
    throw std::runtime_error("The operator MultAT not defined!\n");
}

}
#endif // OPERATOR_HPP__
