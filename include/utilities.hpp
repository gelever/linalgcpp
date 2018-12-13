/*! @file */

#ifndef UTILITIES_HPP__
#define UTILITIES_HPP__

#include <memory>
#include "sparsematrix.hpp"

#if __cplusplus > 201103L
using std::make_unique;
#else
template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&& ... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}
#endif

namespace linalgcpp
{

/*! @brief Throw if false in debug mode only */
inline
void linalgcpp_assert(bool expression, const std::string& message = "linalgcpp assertion failed")
{
#ifndef NDEBUG
    if (!expression)
    {
        throw std::runtime_error(message);
    }
#endif
}

/*! @brief Throw if false unconditionally */
inline
void linalgcpp_verify(bool expression, const std::string& message = "linalgcpp verification failed")
{
    if (!expression)
    {
        throw std::runtime_error(message);
    }
}

/*! @brief Throw if false in debug mode only */
template <typename F>
void linalgcpp_assert(F&& lambda, const std::string& message = "linalgcpp assertion failed")
{
#ifndef NDEBUG
    if (!lambda())
    {
        throw std::runtime_error(message);
    }
#endif
}

/*! @brief Throw if false unconditionally */
template <typename F>
void linalgcpp_verify(F&& lambda, const std::string& message = "linalgcpp verification failed")
{
    if (!lambda())
    {
        throw std::runtime_error(message);
    }
}

/** @brief Sparse identity of given size
    @param size square size of identity
    @return identity matrix
*/
template <typename T>
SparseMatrix<T> SparseIdentity(int size);

/** @brief Construct an rectangular identity matrix (as a SparseMatrix)
    @param rows number of row
    @param cols number of columns
    @param row_offset offset row where diagonal identity starts
    @param col_offset offset column where diagonal identity starts
*/
template <typename T>
SparseMatrix<T> SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

template <typename T = double>
SparseMatrix<T> SparseIdentity(int size)
{
    assert(size >= 0);

    return SparseMatrix<T>(std::vector<T>(size, (T)1.0));
}

template <typename T = double>
SparseMatrix<T> SparseIdentity(int rows, int cols, int row_offset, int col_offset)
{
    assert(rows >= 0);
    assert(cols >= 0);
    assert(row_offset <= rows);
    assert(row_offset >= 0);
    assert(col_offset <= cols);
    assert(col_offset >= 0);

    const int diag_size = std::min(rows - row_offset, cols - col_offset);

    std::vector<int> indptr(rows + 1);

    std::fill(std::begin(indptr), std::begin(indptr) + row_offset, 0);
    std::iota(std::begin(indptr) + row_offset, std::begin(indptr) + row_offset + diag_size, 0);
    std::fill(std::begin(indptr) + row_offset + diag_size, std::begin(indptr) + rows + 1, diag_size);

    std::vector<int> indices(diag_size);
    std::iota(std::begin(indices), std::begin(indices) + diag_size, col_offset);

    std::vector<T> data(diag_size, 1.0);

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

} //namespace linalgcpp

#endif // UTILITIES_HPP__

