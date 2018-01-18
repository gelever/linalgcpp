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
        explicit BlockVector(std::vector<size_t> offsets);

        /*! @brief Constructor with given offsets and values*/
        explicit BlockVector(std::vector<size_t> offsets, T val);

        /*! @brief Constructor with given data and offsets */
        explicit BlockVector(const VectorView<T>& data, std::vector<size_t> offsets);

        const std::vector<size_t>& GetOffsets() const;

        /*! @brief Get a view of a block */
        VectorView<T> GetBlock(size_t block);

        /*! @brief Get a const view of a block
            @note returns const rvalue reference (const VectorView<T>&&)
        */
        const auto GetBlock(size_t block) const;

        using Vector<T>::operator=;
    private:
        std::vector<size_t> offsets_;
};

template <typename T>
BlockVector<T>::BlockVector()
    : offsets_(1, 0)
{

}

template <typename T>
BlockVector<T>::BlockVector(std::vector<size_t> offsets)
    : Vector<T>(offsets.back()), offsets_(offsets)
{

}

template <typename T>
BlockVector<T>::BlockVector(std::vector<size_t> offsets, T val)
    : Vector<T>(offsets.back(), val), offsets_(offsets)
{

}

template <typename T>
BlockVector<T>::BlockVector(const VectorView<T>& data, std::vector<size_t> offsets)
    : Vector<T>(data), offsets_(offsets)
{

}
template <typename T>
const std::vector<size_t>& BlockVector<T>::GetOffsets() const
{
    return offsets_;
}

template <typename T>
VectorView<T> BlockVector<T>::GetBlock(size_t block)
{
    assert(block < offsets_.size() - 1);

    T* data = Vector<T>::begin() + offsets_[block];
    size_t size = offsets_[block + 1] - offsets_[block];

    return VectorView<T> {data, size};
}

template <typename T>
const auto BlockVector<T>::GetBlock(size_t block) const
{
    assert(block < offsets_.size() - 1);

    T* data = const_cast<T*>(Vector<T>::begin() + offsets_[block]);
    size_t size = offsets_[block + 1] - offsets_[block];

    return VectorView<T> {data, size};
}

} //namespace linalgcpp

#endif // BLOCKVECTOR_HPP__
