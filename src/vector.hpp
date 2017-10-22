/* @file */

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

namespace linalgcpp
{

/* @brief Vector represents a mathmatical vector
          with a size and data type
*/
template <typename T = double>
class Vector
{
    public:
        /* @brief Default Constructor of zero size */
        Vector() = default;

        /* @brief Constructor of setting the size
           @param size the length of the vector
        */
        Vector(size_t size);

        /* @brief Constructor of setting the size and intial values
           @param size the length of the vector
           @param val the initial value to set all entries to
        */
        Vector(size_t size, T val);

        /* @brief Constructor from an std::vector*/
        Vector(std::vector<T> vect);

        /* @brief Copy Constructor*/
        Vector(const Vector& vect) = default;

        /* @brief Move constructor */
        Vector(Vector&& vect);

        /* @brief Destructor */
        ~Vector() noexcept = default;

        /* @brief Sets this vector equal to another
           @param other the vector to copy
        */
        Vector& operator=(Vector vect);

        /* @brief Sets all entries to a scalar value
           @param val the value to set all entries to
        */
        Vector& operator=(T val);

        /* @brief Swap two vectors
           @param lhs left hand side vector
           @param rhs right hand side vector
        */
        template <typename T2>
        friend void Swap(Vector<T2>& lhs, Vector<T2>& rhs);

        /* @brief STL like begin. Points to start of data
           @retval pointer to the start of data
        */
        T* begin();

        /* @brief STL like end. Points to the end of data
           @retval pointer to the end of data
        */
        T* end();

        /* @brief STL like const begin. Points to start of data
           @retval const pointer to the start of data
        */
        const T* begin() const;

        /* @brief STL like const end. Points to the end of data
           @retval const pointer to the end of data
        */
        const T* end() const;

        /* @brief Index operator
           @param i index into vector
           @retval reference to value at index i
        */
        T& operator[](int i);

        /* @brief Const index operator
           @param i index into vector
           @retval const reference to value at index i
        */
        const T& operator[](int i) const;

        /* @brief Get the length of the vector
           @retval the length of the vector
        */
        size_t size() const;

        /* @brief Print the vector entries
           @param label the label to print before the list of entries
           @param out stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

    private:
        std::vector<T> data_;

};

template <typename T>
Vector<T>::Vector(size_t size)
{
    data_.resize(size);
}

template <typename T>
Vector<T>::Vector(size_t size, T val)
{
    data_.resize(size, val);
}

template <typename T>
Vector<T>::Vector(std::vector<T> vect)
{
    std::swap(vect, data_);
}

template <typename T>
Vector<T>::Vector(Vector<T>&& vect)
{
    std::swap(*this, vect);
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector<T> vect)
{
    std::swap(*this, vect);

    return *this;
}

template <typename T2>
void Swap(Vector<T2>& lhs, Vector<T2>& rhs)
{
    std::swap(lhs.data_, rhs.data_);
}

template <typename T>
Vector<T>& Vector<T>::operator=(T val)
{
    std::fill(std::begin(data_), std::end(data_), val);

    return *this;
}

template <typename T>
T* Vector<T>::begin()
{
    return data_.data();
}

template <typename T>
T* Vector<T>::end()
{
    return data_.data() + data_.size();
}

template <typename T>
const T* Vector<T>::begin() const
{
    return data_.data();
}

template <typename T>
const T* Vector<T>::end() const
{
    return data_.data() + data_.size();
}

template <typename T>
T& Vector<T>::operator[](int i)
{
    assert(i >= 0);
    assert(static_cast<unsigned int>(i) < data_.size());

    return data_[i];
}

template <typename T>
const T& Vector<T>::operator[](int i) const
{
    assert(i >= 0);
    assert(static_cast<unsigned int>(i) < data_.size());

    return data_[i];
}

template <typename T>
size_t Vector<T>::size() const
{
    return data_.size();
}

template <typename T>
void Vector<T>::Print(const std::string& label, std::ostream& out) const
{
    out << label;

    out << (*this);

    out << "\n";
}


// Inline Vector Free Functions

/* @brief Compute the L2 norm of the vector
   @param vect the vector to compute the L2 norm of
   @retval the L2 norm
*/
template <typename T>
double L2Norm(const Vector<T>& vect)
{
    return std::sqrt(InnerProduct(vect, vect));
}

/* @brief Print the vector to a stream
   @param out stream to print to
   @param vect the vector to print
*/
template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& vect)
{
    out << "\n";

    for (const auto& i : vect)
    {
        out << i << "\n";
    }

    out << "\n";

    return out;
}

/* @brief Compute the inner product two vectors x^T y
   @param lhs left hand side vector x
   @param rhs right hand side vector y
   @retval the inner product
*/
template <typename T, typename T2>
double InnerProduct(const Vector<T>& lhs, const Vector<T2>& rhs)
{
    assert(lhs.size() == rhs.size());

    double start = 0.0;
    return std::inner_product(std::begin(lhs), std::end(lhs), std::begin(rhs), start);
}

/* @brief Compute the inner product two vectors x^T y
   @param lhs left hand side vector x
   @param rhs right hand side vector y
   @retval the inner product
*/
template <typename T, typename T2>
double operator*(const Vector<T>& lhs, const Vector<T2>& rhs)
{
    return InnerProduct(lhs, rhs);
}

/* @brief Entrywise multiplication x_i = x_i * y_i
   @param lhs left hand side vector x
   @param rhs right hand side vector y
*/
template <typename T>
Vector<T>& operator*=(Vector<T>& lhs, const Vector<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const int size = lhs.size();

    for (int i = 0; i < size; ++i)
    {
        lhs[i] *= rhs[i];
    }

    return lhs;
}

/* @brief Entrywise subtraction x_i = x_i - y_i
   @param lhs left hand side vector x
   @param rhs right hand side vector y
*/
template <typename T>
Vector<T>& operator-=(Vector<T>& lhs, const Vector<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const int size = lhs.size();

    for (int i = 0; i < size; ++i)
    {
        lhs[i] -= rhs[i];
    }

    return lhs;
}

/* @brief Multiply a vector by a scalar
   @param vect vector to multiply
   @param val value to scale by
*/
template <typename T, typename T2>
Vector<T>& operator*=(Vector<T>& vect, T2 val)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}

/* @brief Divide a vector by a scalar
   @param vect vector to multiply
   @param val value to scale by
*/
template <typename T, typename T2>
Vector<T>& operator/=(Vector<T>& vect, T2 val)
{
    assert(val != 0);

    for (T& i : vect)
    {
        i /= val;
    }

    return vect;
}

/* @brief Multiply a vector by a scalar into a new vector
   @param vect vector to multiply
   @param val value to scale by
   @retval the vector multiplied by the scalar
*/
template <typename T, typename T2>
Vector<T> operator*(Vector<T> vect, T2 val)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}

/* @brief Multiply a vector by a scalar into a new vector
   @param vect vector to multiply
   @param val value to scale by
   @retval the vector multiplied by the scalar
*/
template <typename T, typename T2>
Vector<T> operator*(T2 val, Vector<T> vect)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}


/* @brief Add two vectors into a new vector z = x + y
   @param lhs left hand side vector x
   @param rhs right hand side vector y
   @retval the sum of the two vectors
*/
template <typename T>
Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
{
    return lhs += rhs;
}

/* @brief Subtract two vectors into a new vector z = x - y
   @param lhs left hand side vector x
   @param rhs right hand side vector y
   @retval the difference of the two vectors
*/
template <typename T>
Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs)
{
    return lhs -= rhs;
}

/* @brief Add a scalar to each entry
   @param lhs vector to add to
   @param val the value to add
*/
template <typename T, typename T2>
Vector<T>& operator+=(Vector<T>& lhs, T2 val)
{
    const int size = lhs.size();

    for (T& i : lhs)
    {
        i += val;
    }

    return lhs;
}

/* @brief Subtract a scalar from each entry
   @param lhs vector to add to
   @param val the value to subtract
*/
template <typename T, typename T2>
Vector<T>& operator-=(Vector<T>& lhs, T2 val)
{
    const int size = lhs.size();

    for (T& i : lhs)
    {
        i -= val;
    }

    return lhs;
}


/* @brief Compute the maximum entry value in a vector
   @param vect vector to find the max
   @retval the maximum entry value
*/
template <typename T>
T Max(const Vector<T>& vect)
{
    return *std::max_element(std::begin(vect), std::end(vect));
}

/* @brief Compute the minimum entry value in a vector
   @param vect vector to find the minimum
   @retval the minimum entry value
*/
template <typename T>
T Min(const Vector<T>& vect)
{
    return *std::min_element(std::begin(vect), std::end(vect));
}

/* @brief Compute the sum of all vector entries
   @param vect vector to find the sum
   @retval the sum of all entries
*/
template <typename T>
T Sum(const Vector<T>& vect)
{
    T total = 0.0;
    std::accumulate(std::begin(vect), std::end(vect), total);

    return total;
}

/* @brief Compute the mean of all vector entries
   @param vect vector to find the mean
   @retval the mean of all entries
*/
template <typename T>
double Mean(const Vector<T>& vect)
{
    return Sum(vect) / vect.size();
}

/* @brief Print an std vector to a stream
   @param out stream to print to
   @param vect the vector to print
*/
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vect)
{
    out << "\n";

    for (const auto& i : vect)
    {
        out << i << "\n";
    }

    out << "\n";

    return out;
}

/* @brief Randomize the entries in a double vector
   @param vect vector to randomize
   @param lo lower range limit
   @param hi upper range limit
*/
void Randomize(Vector<double>& vect, double lo = 0.0, double hi = 1.0);

/* @brief Randomize the entries in a integer vector
   @param vect vector to randomize
   @param lo lower range limit
   @param hi upper range limit
*/
void Randomize(Vector<int>& vect, int lo = 0.0, int hi = 1.0);

/* @brief Normalize a vector such that its L2 norm is 1.0
   @param vect vector to normalize
*/
void Normalize(Vector<double>& vect);

/* @brief Subtract a constant vector set to the average 
   from this vector: x_i = x_i - mean(x)
   @param vect vector to subtract average from
*/
void SubAvg(Vector<double>& vect);

} // namespace linalgcpp

#endif // VECTOR_HPP__
