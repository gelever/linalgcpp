/*! @file */

#ifndef PARBLOCKMATRIX_HPP
#define PARBLOCKMATRIX_HPP

#include "linalgcpp.hpp"
#include "parmatrix.hpp"

namespace linalgcpp
{

/*! @brief Dense parallel block matrix where each block is a parallel sparse matrix
           Blocks of zero size are assumed to be zero matrix
           Offsets are to be of size (blocks + 1), where the last
           entry is the total size.

           @warning This is NOT a view.  BlockMatrix copies/takes
           possession of all given input!
*/
class ParBlockMatrix : public ParOperator
{
    public:
         /*! @brief Default Constructor of zero size */
         ParBlockMatrix();

         /*! @brief Square Constructor with given symmetric offsets*/
         ParBlockMatrix(MPI_Comm comm, std::vector<int> offsets);

         /*! @brief Rectangle Constructor with given offsets*/
         ParBlockMatrix(MPI_Comm comm, std::vector<int> row_offsets, std::vector<int> col_offsets);

         /*! @brief Copy deconstructor */
         ParBlockMatrix(const ParBlockMatrix& other) noexcept;

         /*! @brief Move deconstructor */
         ParBlockMatrix(ParBlockMatrix&& other) noexcept;

         /*! @brief Assignment operator */
         ParBlockMatrix& operator=(ParBlockMatrix other) noexcept;

         friend void swap(ParBlockMatrix& lhs, ParBlockMatrix& rhs) noexcept;

         /*! @brief Default deconstructor */
         ~ParBlockMatrix() noexcept = default;

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
             @retval ParMatrix at index (i, j)
         */
         const ParMatrix& GetBlock(int i, int j) const;

         /*! @brief Set a block
             @param i row index
             @param j column index
             @param ParMatrix at index (i, j)
         */
         void SetBlock(int i, int j, ParMatrix mat);

         /*! @brief Get the transpose matrix
         */
         ParBlockMatrix Transpose() const;

         /*! @brief Multiply another ParBlockMatrix
         */
         //ParBlockMatrix Mult(const ParBlockMatrix& rhs) const;

         /*! @brief Get the combined monolithic matrix
         */
         //ParMatrix Combine() const;

         /// Operator Requirement
         void Mult(const VectorView<double>& input, VectorView<double> output) const override;
         /// Operator Requirement
         void MultAT(const VectorView<double>& input, VectorView<double> output) const override;

         using Operator::Mult;

    private:
        std::vector<int> row_offsets_;
        std::vector<int> col_offsets_;

        std::vector<std::vector<ParMatrix>> A_;
};

} //namespace linalgcpp

#endif // BLOCKMATRIX_HPP
