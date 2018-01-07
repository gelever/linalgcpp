/*! @file */

#ifndef VECTOR_HPP__
#define VECTOR_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <assert.h>

#include "vectorview.hpp"

namespace linalgcpp
{

template <typename T = double>
class Vector : public VectorView<T>
{
    public:
        /*! @brief Default Constructor of zero size */
        Vector() = default;

        /*! @brief Constructor of setting the size
            @param size the length of the vector
        */
        explicit Vector(size_t size);

        /*! @brief Constructor of setting the size and intial values
            @param size the length of the vector
            @param val the initial value to set all entries to
        */
        Vector(size_t size, T val);

        /*! @brief Constructor from view
            @note deep copy
         * */
        explicit Vector(const VectorView<T>& vect);

        /*! @brief Constructor from an std::vector*/
        explicit Vector(std::vector<T> vect);

        /*! @brief Copy Constructor */
        Vector(const Vector& vect) noexcept;

        /*! @brief Move constructor
            @param vect the vector to move
        */
        Vector(Vector&& vect) noexcept;

        /*! @brief Destructor
        */
        ~Vector() noexcept = default;

        /*! @brief Sets this vector equal to another
            @param vect the vector to copy
        */

        Vector& operator=(const Vector<T>& vect) noexcept;

        /*! @brief Sets this vector equal to another
            @param vect the vector to copy
        */
        Vector& operator=(const VectorView<T>& vect) noexcept;

        /*! @brief Sets this vector equal to another
            @param vect the vector to copy
        */
        Vector& operator=(Vector&& vect) noexcept;

        /*! @brief Swap two vectors
            @param lhs left hand side vector
            @param rhs right hand side vector
        */
        template <typename T2>
        friend void Swap(Vector<T2>& lhs, Vector<T2>& rhs);

        using VectorView<T>::operator=;

    private:
        std::vector<T> data_;

};

template <typename T>
Vector<T>::Vector(size_t size)
{
    data_.resize(size);

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);
}

template <typename T>
Vector<T>::Vector(size_t size, T val)
{
    data_.resize(size, val);

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);
}

template <typename T>
Vector<T>::Vector(const VectorView<T>& vect)
{
    data_.resize(vect.size());
    std::copy(std::begin(vect), std::end(vect), std::begin(data_));

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);
}

template <typename T>
Vector<T>::Vector(std::vector<T> vect)
{
    std::swap(vect, data_);

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);
}

template <typename T>
Vector<T>::Vector(const Vector<T>& vect) noexcept
    : data_(vect.data_)
{
    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);
}

template <typename T>
Vector<T>::Vector(Vector<T>&& vect) noexcept
{
    Swap(*this, vect);

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);
}

template <typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& vect) noexcept
{
    data_.resize(vect.size());
    std::copy(std::begin(vect.data_), std::end(vect.data_), std::begin(data_));

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);

    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(const VectorView<T>& vect) noexcept
{
    data_.resize(vect.size());
    std::copy(std::begin(vect), std::end(vect), std::begin(data_));

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);

    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector<T>&& vect) noexcept
{
    Swap(*this, vect);

    VectorView<T> set_view(data_.data(), data_.size());
    VectorView<T>::operator=(set_view);

    return *this;
}

template <typename T2>
void Swap(Vector<T2>& lhs, Vector<T2>& rhs)
{
    Swap(static_cast<VectorView<T2>&>(lhs), static_cast<VectorView<T2>&>(rhs));
    std::swap(lhs.data_, rhs.data_);
}

/*! @brief Multiply a vector by a scalar into a new vector
    @param vect vector to multiply
    @param val value to scale by
    @retval the vector multiplied by the scalar
*/
template <typename T>
Vector<T> operator*(Vector<T> vect, T val)
{
    vect *= val;
    return vect;
}

/*! @brief Multiply a vector by a scalar into a new vector
    @param vect vector to multiply
    @param val value to scale by
    @retval the vector multiplied by the scalar
*/
template <typename T>
Vector<T> operator*(T val, Vector<T> vect)
{
    vect *= val;
    return vect;
}

/*! @brief Divide a vector by a scalar into a new vector
    @param vect vector to divide
    @param val value to scale by
    @retval the vector divided by the scalar
*/
template <typename T>
Vector<T> operator/(Vector<T> vect, T val)
{
    vect /= val;
    return vect;
}

/*! @brief Divide a scalar by vector entries into a new vector
    @param vect vector to multiply
    @param val value to scale by
    @retval the vector multiplied by the scalar
*/
template <typename T>
Vector<T> operator/(T val, Vector<T> vect)
{
    for (T& i : vect)
    {
        i = val / i;
    }

    return vect;
}

/*! @brief Add two vectors into a new vector z = x + y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the sum of the two vectors
*/
template <typename T>
Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

/*! @brief Subtract two vectors into a new vector z = x - y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the difference of the two vectors
*/
template <typename T>
Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

} // namespace linalgcpp

#endif // VECTOR_HPP__
