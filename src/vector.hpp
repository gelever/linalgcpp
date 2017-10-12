#ifndef VECTOR_HPP__
#define VECTOR_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <assert.h>

void Normalize(std::vector<double>& vect);
void SubAvg(std::vector<double>& vect);
double L2Norm(const std::vector<double>& vect);

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vect)
{
	out <<"\n";

	for (const auto& i : vect)
	{
		out << i << "\n";
	}

	out <<"\n";

	return out;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& lhs, const std::vector<T>& rhs)
{
	assert(lhs.size() == rhs.size());

	const int size = lhs.size();

	for (int i = 0; i < size; ++i)
	{
		lhs[i] += rhs[i];
	}

	return lhs;
}

template <typename T>
std::vector<T>& operator-=(std::vector<T>& lhs, const std::vector<T>& rhs)
{
	assert(lhs.size() == rhs.size());

	const int size = lhs.size();

	for (int i = 0; i < size; ++i)
	{
		lhs[i] -= rhs[i];
	}

	return lhs;
}

template <typename T>
std::vector<T> operator+(std::vector<T> lhs, const std::vector<T>& rhs)
{
	return lhs += rhs;
}

template <typename T>
std::vector<T> operator-(std::vector<T> lhs, const std::vector<T>& rhs)
{
	return lhs -= rhs;
}

template <typename T>
double operator*(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
	assert(lhs.size() == rhs.size());

	T start = 0.0;
	return std::inner_product(begin(lhs), end(lhs), begin(rhs), start);
}

template <typename T, typename T2>
std::vector<T>& operator*=(std::vector<T>& lhs, T2 val)
{
	for (T& i : lhs)
	{
		i *= val;
	}

	return lhs;
}

template <typename T, typename T2>
std::vector<T> operator*(std::vector<T> lhs, T2 val)
{
	return lhs *= val;
}

template <typename T, typename T2>
std::vector<T> operator*(T2 val, std::vector<T> rhs)
{
	return rhs *= val;
}

template <typename T, typename T2>
std::vector<T>& operator/=(std::vector<T>& lhs, T2 val)
{
	assert(val != 0.0);

	for (T& i : lhs)
	{
		i /= val;
	}

	return lhs;
}

template <typename T, typename T2>
std::vector<T>& operator+=(std::vector<T>& lhs, T2 val)
{
	assert(val != 0.0);

	for (T& i : lhs)
	{
		i += val;
	}

	return lhs;
}

template <typename T, typename T2>
std::vector<T>& operator-=(std::vector<T>& lhs, T2 val)
{
	assert(val != 0.0);

	for (T& i : lhs)
	{
		i -= val;
	}

	return lhs;
}

template <typename T>
T Max(const std::vector<T>& vect)
{
	return *std::max_element(begin(vect), end(vect));
}

template <typename T>
T Min(const std::vector<T>& vect)
{
	return *std::min_element(begin(vect), end(vect));
}

template <typename T>
T Sum(const std::vector<T>& vect)
{
	T total = 0.0;
	std::accumulate(begin(vect), end(vect), total);

	return total;
}

template <typename T>
T Mean(const std::vector<T>& vect)
{
	return Sum(vect) / vect.size();
}



#endif // VECTOR_HPP__
