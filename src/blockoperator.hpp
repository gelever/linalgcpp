/*! @file */

#ifndef BLOCKOPERATOR_HPP__
#define BLOCKOPERATOR_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

#include "operator.hpp"
#include "blockvector.hpp"

namespace linalgcpp
{

/*! @brief Dense block matrix where each block is a sparse matrix
           Blocks of zero size are assumed to be zero matrix
           Offsets are to be of size (blocks + 1), where the last
           entry is the total size.

           @warning This is NOT a view.  BlockMatrix copies/takes
           possession of all given input!
*/
class BlockOperator : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        BlockOperator();

        /*! @brief Square Constructor with given symmetric offsets*/
        explicit BlockOperator(std::vector<size_t> offsets);

        /*! @brief Rectangle Constructor with given offsets*/
        BlockOperator(std::vector<size_t> row_offsets, std::vector<size_t> col_offsets);

        /*! @brief Default deconstructor */
        ~BlockOperator() noexcept = default;

        /*! @brief The number of rows in this matrix
            @retval the number of rows
        */
        size_t Rows() const override;

        /*! @brief The number of columns in this matrix
            @retval the number of columns
        */
        size_t Cols() const override;

        /*! @brief Get the row offsets
            @retval the row offsets
        */
        const std::vector<size_t>& GetRowOffsets() const;

        /*! @brief Get the col offsets
            @retval the col offsets
        */
        const std::vector<size_t>& GetColOffsets() const;

        /*! @brief Get a block
            @param i row index
            @param j column index
            @retval SparseMatrix at index (i, j)
        */
        const Operator* GetBlock(size_t i, size_t j) const;

        /*! @brief Set a block
            @param i row index
            @param j column index
            @param SparseMatrix at index (i, j)
        */
        void SetBlock(size_t i, size_t j, const Operator& op);

        /*! @brief Get the transpose matrix
        */
        BlockOperator Transpose() const;

        /// Operator Requirement
        void Mult(const VectorView<double>& input, VectorView<double>& output) const override;

        /// Operator Requirement
        void MultAT(const VectorView<double>& input, VectorView<double>& output) const override;

        using Operator::Mult;

    private:
        std::vector<size_t> row_offsets_;
        std::vector<size_t> col_offsets_;

        std::vector<std::vector<const Operator*>> A_;

        size_t rows_;
        size_t cols_;

        mutable BlockVector<double> x_;
        mutable BlockVector<double> y_;
};

} //namespace linalgcpp

#endif // BLOCKOPERATOR_HPP__
