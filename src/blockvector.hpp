/*! @file */

#ifndef BLOCKVECTOR_HPP__
#define BLOCKVECTOR_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

#include "operator.hpp"
#include "vector.hpp"

namespace linalgcpp
{

/*! @brief Block vector. A vector that knows its offsets.
*/
template <typename T = double>
class BlockVector : public Vector<T>
{
    public:
        /*! @brief Default Constructor of zero size */
        BlockVector();

        /*! @brief Constructor with given offsets */
        explicit BlockVector(std::vector<int> offsets);

        /*! @brief Constructor with given offsets and values*/
        explicit BlockVector(std::vector<int> offsets, T val);

        /*! @brief Constructor with given data and offsets */
        explicit BlockVector(const VectorView<T>& data, std::vector<int> offsets);

        const std::vector<int>& GetOffsets() const;

        /*! @brief Get a view of a block */
        VectorView<T> GetBlock(int block);

        /*! @brief Get a const view of a block
            @note returns const rvalue reference (const VectorView<T>&&)
        */
        const auto GetBlock(int block) const;

        using Vector<T>::operator=;
    private:
        std::vector<int> offsets_;
};

template <typename T>
BlockVector<T>::BlockVector()
    : offsets_(1, 0)
{

}

template <typename T>
BlockVector<T>::BlockVector(std::vector<int> offsets)
    : Vector<T>(offsets.back()), offsets_(offsets)
{

}

template <typename T>
BlockVector<T>::BlockVector(std::vector<int> offsets, T val)
    : Vector<T>(offsets.back(), val), offsets_(offsets)
{

}

template <typename T>
BlockVector<T>::BlockVector(const VectorView<T>& data, std::vector<int> offsets)
    : Vector<T>(data), offsets_(offsets)
{

}
template <typename T>
const std::vector<int>& BlockVector<T>::GetOffsets() const
{
    return offsets_;
}

template <typename T>
VectorView<T> BlockVector<T>::GetBlock(int block)
{
    assert(block < static_cast<int>(offsets_.size()) - 1);

    T* data = Vector<T>::begin() + offsets_[block];
    int size = offsets_[block + 1] - offsets_[block];

    return VectorView<T> {data, size};
}

template <typename T>
const auto BlockVector<T>::GetBlock(int block) const
{
    assert(block < static_cast<int>(offsets_.size()) - 1);

    T* data = const_cast<T*>(Vector<T>::begin() + offsets_[block]);
    int size = offsets_[block + 1] - offsets_[block];

    return VectorView<T> {data, size};
}

} //namespace linalgcpp

#endif // BLOCKVECTOR_HPP__
