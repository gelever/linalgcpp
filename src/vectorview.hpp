/*! @file */

#ifndef VECTORVIEW_HPP__
#define VECTORVIEW_HPP__

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

/*! @brief Vector view of data and size

    @note Views are only modifiable if you own the view
          that you plan to change it to.

          If you want view A to be equal to view B,
          you most own both A and B. Otherwise,
          it is trivial to subvert const restrictions.
          I'm not sure if this is good way or not to
          deal w/ this.
*/
template <typename T = double>
class VectorView
{
    public:
        /*! @brief Default Constructor of zero size */
        VectorView();

        /*! @brief Constructor of with data */
        VectorView(T* data, size_t size);

        /*! @brief Copy Constructor */
        VectorView(VectorView& vect) noexcept;

        /*! @brief Move Constructor */
        VectorView(VectorView&& vect) noexcept;

        /*! @brief Assignment Constructor */
        VectorView& operator=(VectorView& vect) noexcept;

        /*! @brief Destructor */
        ~VectorView() noexcept = default;

        /*! @brief Swap two vectors
            @param lhs left hand side vector
            @param rhs right hand side vector
        */
        template <typename T2>
        friend void Swap(VectorView<T2>& lhs, VectorView<T2>& rhs);

        /*! @brief STL like begin. Points to start of data
            @retval pointer to the start of data
        */
        virtual T* begin();

        /*! @brief STL like end. Points to the end of data
            @retval pointer to the end of data
        */
        virtual T* end();

        /*! @brief STL like const begin. Points to start of data
            @retval const pointer to the start of data
        */
        virtual const T* begin() const;

        /*! @brief STL like const end. Points to the end of data
            @retval const pointer to the end of data
        */
        virtual const T* end() const;

        /*! @brief Index operator
            @param i index into vector
            @retval reference to value at index i
        */
        virtual T& operator[](size_t i);

        /*! @brief Const index operator
            @param i index into vector
            @retval const reference to value at index i
        */
        virtual const T& operator[](size_t i) const;

        /*! @brief Get the length of the vector
            @retval the length of the vector
        */
        virtual size_t size() const;

        /*! @brief Sets all entries to a scalar value
            @param val the value to set all entries to
        */
        VectorView& operator=(T val);

        /*! @brief Print the vector entries
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        virtual void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Inner product of two vectors
            @param vect other vector
        */
        virtual T Mult(const VectorView<T>& vect) const;

        /*! @brief Add (alpha * vect) to this vector
            @param alpha scale of rhs
            @param vect vector to add
        */
        VectorView<T>& Add(double alpha, const VectorView<T>& vect);

        /*! @brief Add vect to this vector
            @param vect vector to add
        */
        VectorView<T>& Add(const VectorView<T>& vect);

        /*! @brief Subtract (alpha * vect) from this vector
            @param alpha scale of rhs
            @param vect vector to subtract
        */
        VectorView<T>& Sub(double alpha, const VectorView<T>& vect);

        /*! @brief Subtract vect from this vector
            @param vect vector to subtract
        */
        VectorView<T>& Sub(const VectorView<T>& vect);

        /*! @brief Compute the L2 norm of the vector
            @retval the L2 norm
        */
        virtual double L2Norm() const;

    protected:
        //void SetData(T* data, size_t size);

    private:
        T* data_;
        size_t size_;
};

template <typename T>
VectorView<T>::VectorView()
    : data_(nullptr), size_(0)
{

}

template <typename T>
VectorView<T>::VectorView(T* data, size_t size)
    : data_(data), size_(size)
{

}

template <typename T>
VectorView<T>::VectorView(VectorView<T>& vect) noexcept
    : data_(vect.data_), size_(vect.size_)
{

}

template <typename T>
VectorView<T>::VectorView(VectorView<T>&& vect) noexcept
{
    Swap(*this, vect);
}

template <typename T>
VectorView<T>& VectorView<T>::operator=(VectorView<T>& vect) noexcept
{
    data_ = vect.data_;
    size_ = vect.size_;

    return *this;
}

template <typename T2>
void Swap(VectorView<T2>& lhs, VectorView<T2>& rhs)
{
    std::swap(lhs.data_, rhs.data_);
    std::swap(lhs.size_, rhs.size_);
}

template <typename T>
VectorView<T>& VectorView<T>::operator=(T val)
{
    std::fill(begin(), end(), val);

    return *this;
}

template <typename T>
T* VectorView<T>::begin()
{
    return data_;
}

template <typename T>
T* VectorView<T>::end()
{
    return data_ + size_;
}

template <typename T>
const T* VectorView<T>::begin() const
{
    return data_;
}

template <typename T>
const T* VectorView<T>::end() const
{
    return data_ + size_;
}

template <typename T>
T& VectorView<T>::operator[](size_t i)
{
    assert(i < size_);

    return data_[i];
}

template <typename T>
const T& VectorView<T>::operator[](size_t i) const
{
    assert(i < size_);

    return data_[i];
}

template <typename T>
size_t VectorView<T>::size() const
{
    return size_;
}

template <typename T>
void VectorView<T>::Print(const std::string& label, std::ostream& out) const
{
    out << label << "\n";

    for (size_t i = 0; i < size_; ++i)
    {
        out << data_[i] << "\n";
    }

    out << "\n";
}

template <typename T>
VectorView<T>& VectorView<T>::Add(double alpha, const VectorView<T>& rhs)
{
    assert(rhs.size_ == size_);

    for (size_t i = 0; i < size_; ++i)
    {
        data_[i] += alpha * rhs[i];
    }

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Add(const VectorView<T>& rhs)
{
    (*this) += rhs;

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Sub(double alpha, const VectorView<T>& rhs)
{
    assert(rhs.size_ == size_);

    for (size_t i = 0; i < size_; ++i)
    {
        data_[i] -= alpha * rhs[i];
    }

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Sub(const VectorView<T>& rhs)
{
    (*this) -= rhs;

    return *this;
}

/*! @brief Compute the inner product two vectors x^T y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the inner product
*/
template <typename T, typename T2>
double InnerProduct(const VectorView<T>& lhs, const VectorView<T2>& rhs)
{
    return lhs.Mult(rhs);
}

template <typename T>
double VectorView<T>::L2Norm() const
{
    return std::sqrt(InnerProduct(*this, *this));
}

template <typename T>
T VectorView<T>::Mult(const VectorView<T>& vect) const
{
    assert(vect.size() == size());

    T start = 0.0;

    return std::inner_product(begin(), end(), std::begin(vect), start);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const VectorView<T>& vect)
{
    std::string label = "";
    vect.Print(label, out);

    return out;
}

// Templated Free Functions
/*! @brief Compute the L2 norm of the vector
    @param vect the vector to compute the L2 norm of
    @retval the L2 norm
*/
template <typename T>
double L2Norm(const VectorView<T>& vect)
{
    return vect.L2Norm();
}

/*! @brief Compute the inner product two vectors x^T y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the inner product
*/
template <typename T, typename T2>
double operator*(const VectorView<T>& lhs, const VectorView<T2>& rhs)
{
    return InnerProduct(lhs, rhs);
}

/*! @brief Entrywise multiplication x_i = x_i * y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T>
VectorView<T>& operator*=(VectorView<T>& lhs, const VectorView<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const size_t size = lhs.size();

    for (size_t i = 0; i < size; ++i)
    {
        lhs[i] *= rhs[i];
    }

    return lhs;
}

/*! @brief Entrywise division x_i = x_i / y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T>
VectorView<T>& operator/=(VectorView<T>& lhs, const VectorView<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const size_t size = lhs.size();

    for (size_t i = 0; i < size; ++i)
    {
        assert(rhs[i] != 0.0);

        lhs[i] /= rhs[i];
    }

    return lhs;
}

/*! @brief Entrywise addition x_i = x_i - y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T, typename T2>
VectorView<T>& operator+=(VectorView<T>& lhs, const VectorView<T2>& rhs)
{
    assert(lhs.size() == rhs.size());

    const size_t size = lhs.size();

    for (size_t i = 0; i < size; ++i)
    {
        lhs[i] += rhs[i];
    }

    return lhs;
}

/*! @brief Entrywise subtraction x_i = x_i - y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T>
VectorView<T>& operator-=(VectorView<T>& lhs, const VectorView<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const size_t size = lhs.size();

    for (size_t i = 0; i < size; ++i)
    {
        lhs[i] -= rhs[i];
    }

    return lhs;
}

/*! @brief Multiply a vector by a scalar
    @param vect vector to multiply
    @param val value to scale by
*/
template <typename T>
VectorView<T>& operator*=(VectorView<T>& vect, T val)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}

/*! @brief Divide a vector by a scalar
    @param vect vector to multiply
    @param val value to scale by
*/
template <typename T>
VectorView<T>& operator/=(VectorView<T>& vect, T val)
{
    assert(val != 0);

    for (T& i : vect)
    {
        i /= val;
    }

    return vect;
}

/*! @brief Add a scalar to each entry
    @param lhs vector to add to
    @param val the value to add
*/
template <typename T>
VectorView<T>& operator+=(VectorView<T>& lhs, T val)
{
    for (T& i : lhs)
    {
        i += val;
    }

    return lhs;
}

/*! @brief Subtract a scalar from each entry
    @param lhs vector to add to
    @param val the value to subtract
*/
template <typename T>
VectorView<T>& operator-=(VectorView<T>& lhs, T val)
{
    for (T& i : lhs)
    {
        i -= val;
    }

    return lhs;
}

/*! @brief Check if two vectors are equal
    @param lhs left hand side vector
    @param rhs right hand side vector
    @retval true if vectors are close enough to equal
*/
template <typename T, typename T2>
bool operator==(const VectorView<T>& lhs, const VectorView<T2>& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    const size_t size = lhs.size();
    constexpr double tol = 1e-12;

    for (size_t i = 0; i < size; ++i)
    {
        if (std::fabs(lhs[i] - rhs[i]) > tol)
        {
            return false;
        }
    }

    return true;
}

/*! @brief Compute the absolute value maximum entry in a vector
    @param vect vector to find the max
    @retval the maximum entry value
*/
template <typename T>
T AbsMax(const VectorView<T>& vect)
{
    const auto compare = [](auto lhs, auto rhs)
    {
        return std::fabs(lhs) < std::fabs(rhs);
    };

    return std::fabs(*std::max_element(std::begin(vect), std::end(vect), compare));
}

/*! @brief Compute the maximum entry value in a vector
    @param vect vector to find the max
    @retval the maximum entry value
*/
template <typename T>
T Max(const VectorView<T>& vect)
{
    return *std::max_element(std::begin(vect), std::end(vect));
}

/*! @brief Compute the minimum entry value in a vector
    @param vect vector to find the minimum
    @retval the minimum entry value
*/
template <typename T>
T Min(const VectorView<T>& vect)
{
    return *std::min_element(std::begin(vect), std::end(vect));
}

/*! @brief Compute the absolute value minimum entry in a vector
    @param vect vector to find the minimum
    @retval the minimum entry value
*/
template <typename T>
T AbsMin(const VectorView<T>& vect)
{
    const auto compare = [](auto lhs, auto rhs)
    {
        return std::fabs(lhs) < std::fabs(rhs);
    };

    return std::fabs(*std::min_element(std::begin(vect), std::end(vect), compare));
}

/*! @brief Compute the sum of all vector entries
    @param vect vector to find the sum
    @retval the sum of all entries
*/
template <typename T>
T Sum(const VectorView<T>& vect)
{
    T start = 0;
    return std::accumulate(std::begin(vect), std::end(vect), start);
}

/*! @brief Compute the mean of all vector entries
    @param vect vector to find the mean
    @retval the mean of all entries
*/
template <typename T>
double Mean(const VectorView<T>& vect)
{
    return Sum(vect) / vect.size();
}

/*! @brief Print an std vector to a stream
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

/*! @brief Randomize the entries in a double vector
    @param vect vector to randomize
    @param lo lower range limit
    @param hi upper range limit
    @param seed seed to rng, if positive
*/
void Randomize(VectorView<double>& vect, double lo = 0.0, double hi = 1.0, int seed = -1);

/*! @brief Randomize the entries in a integer vector
    @param vect vector to randomize
    @param lo lower range limit
    @param hi upper range limit
    @param seed seed to rng, if positive
*/
void Randomize(VectorView<int>& vect, int lo = 0, int hi = 1, int seed = -1);

/*! @brief Normalize a vector such that its L2 norm is 1.0
    @param vect vector to normalize
*/
void Normalize(VectorView<double>& vect);

/*! @brief Subtract a constant vector set to the average
    from this vector: x_i = x_i - mean(x)
    @param vect vector to subtract average from
*/
void SubAvg(VectorView<double>& vect);

} // namespace linalgcpp

#endif // VECTORVIEW_HPP
