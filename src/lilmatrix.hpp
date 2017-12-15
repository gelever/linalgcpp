/*! @file */

#ifndef LILMATRIX_HPP__
#define LILMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <queue>
#include <tuple>
#include <list>
#include <assert.h>

#include "sparsematrix.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

/*! @brief Linked List Matrix that keeps track
           of individual entries in a matrix.

    @note Multiple entries for a single coordinate
    are summed together
*/
template <typename T>
class LilMatrix : public Operator
{
    public:

        /*! @brief Get the number of rows.
            @retval the number of rows

            @note If the matrix size was not specified during
            creation, then the number of rows is determined
            by the maximum element.
        */
        size_t Rows() const override { return rows_;};

        /*! @brief Get the number of columns.
            @retval the number of columns

            @note If the matrix size was not specified during
            creation, then the number of columns is determined
            by the maximum element
        */
        size_t Cols() const override { return cols_;};

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        void Mult(const Vector<double>& input, Vector<double>& output) const override {};

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        void MultAT(const Vector<double>& input, Vector<double>& output) const override {};

    private:
        size_t rows_;
        size_t cols_;

        std::vector<std::list<std::tuple<int, T>>> entries_;

};



} // namespace linalgcpp

#endif // LILMATRIX_HPP__
