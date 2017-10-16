#ifndef VECTOR_HPP__
#define VECTOR_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <assert.h>

namespace linalgcpp
{

template <typename T = double>
class Vector
{
    public:
        Vector() = default;
        Vector(int size);
        Vector(int size, T val);

        Vector(const Vector& vect) = default;
        ~Vector() noexcept = default;

        Vector(Vector&& vect);
        Vector& operator=(Vector vect);
        Vector& operator=(T val);

        T* begin();
        T* end();

        const T* begin() const;
        const T* end() const;

        T& operator[](int i);
        const T& operator[](int i) const;

        int size() const
        {
            return data_.size();
        }

    private:
        std::vector<T> data_;

};

template <typename T>
Vector<T>::Vector(int size)
{
    assert(size > 0);

    data_.resize(size);
}

template <typename T>
Vector<T>::Vector(int size, T val)
{
    assert(size > 0);

    data_.resize(size, val);
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
double L2Norm(const Vector<T>& vect)
{
    return std::sqrt(InnerProduct(vect, vect));
}

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

template <typename T, typename T2>
double InnerProduct(const Vector<T>& lhs, const Vector<T2>& rhs)
{
    assert(lhs.size() == rhs.size());

    double start = 0.0;
    return std::inner_product(std::begin(lhs), std::end(lhs), std::begin(rhs), start);
}

template <typename T, typename T2>
double operator*(const Vector<T>& lhs, const Vector<T2>& rhs)
{
    return InnerProduct(lhs, rhs);
}

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

template <typename T, typename T2>
Vector<T>& operator*=(Vector<T>& vect, T2 val)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}

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

template <typename T, typename T2>
Vector<T> operator*(Vector<T> vect, T2 val)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}

template <typename T, typename T2>
Vector<T> operator*(T2 val, Vector<T> vect)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}


template <typename T>
Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
{
    return lhs += rhs;
}

template <typename T>
Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs)
{
    return lhs -= rhs;
}

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


template <typename T>
T Max(const Vector<T>& vect)
{
    return *std::max_element(std::begin(vect), std::end(vect));
}

template <typename T>
T Min(const Vector<T>& vect)
{
    return *std::min_element(std::begin(vect), std::end(vect));
}

template <typename T>
T Sum(const Vector<T>& vect)
{
    T total = 0.0;
    std::accumulate(std::begin(vect), std::end(vect), total);

    return total;
}

template <typename T>
double Mean(const Vector<T>& vect)
{
    return Sum(vect) / vect.size();
}

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

void Normalize(Vector<double>& vect);
void SubAvg(Vector<double>& vect);

} // namespace linalgcpp

#endif // VECTOR_HPP__
