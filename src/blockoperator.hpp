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

           @Note This is a view.  BlockOperator does not take
           possession of any given input!
*/
class BlockOperator : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        BlockOperator();

        /*! @brief Square Constructor with given symmetric offsets*/
        explicit BlockOperator(std::vector<int> offsets);

        /*! @brief Rectangle Constructor with given offsets*/
        BlockOperator(std::vector<int> row_offsets, std::vector<int> col_offsets);

        /*! @brief Copy deconstructor */
        BlockOperator(const BlockOperator& other) noexcept;

        /*! @brief Move deconstructor */
        BlockOperator(BlockOperator&& other) noexcept;

        /*! @brief Default deconstructor */
        ~BlockOperator() noexcept = default;

        /*! @brief Swap two block operators */
        friend void swap(BlockOperator& lhs, BlockOperator& rhs) noexcept;

        /*! @brief Get the row offsets
            @retval the row offsets
        */
        const std::vector<int>& GetRowOffsets() const;

        /*! @brief Get the col offsets
            @retval the col offsets
        */
        const std::vector<int>& GetColOffsets() const;

        /*! @brief Get a block
            @param i row index
            @param j column index
            @retval SparseMatrix at index (i, j)
        */
        const Operator* GetBlock(int i, int j) const;

        /*! @brief Set a block
            @param i row index
            @param j column index
            @param SparseMatrix at index (i, j)
        */
        void SetBlock(int i, int j, const Operator& op);

        /*! @brief Get the transpose matrix
        */
        BlockOperator Transpose() const;

        /// Operator Requirement
        void Mult(const VectorView<double>& input, VectorView<double>& output) const override;

        /// Operator Requirement
        void MultAT(const VectorView<double>& input, VectorView<double>& output) const override;

        using Operator::Mult;

    private:
        std::vector<int> row_offsets_;
        std::vector<int> col_offsets_;

        std::vector<std::vector<const Operator*>> A_;

        mutable BlockVector<double> x_;
        mutable BlockVector<double> y_;
};

} //namespace linalgcpp

#endif // BLOCKOPERATOR_HPP__
