/*! @file */

#ifndef UTILITIES_HPP__
#define UTILITIES_HPP__

#include <memory>
#include "sparsematrix.hpp"
#include "coomatrix.hpp"

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

/** @brief Set a global marker with given local indices
    Such that marker[global_index] = local_index and
    all other entries are -1

    @param marker global marker to set
    @param indices local indices
*/
void SetMarker(std::vector<int>& marker, const std::vector<int>& indices);

/** @brief Clear a global marker with given local indices
    Such that marker[global_index] = -1

    @param marker global marker to clear
    @param indices local indices
*/
void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices);


/** @brief Sparse identity of given size
    @param size square size of identity
    @return identity matrix
*/
template <typename T = double>
SparseMatrix<T> SparseIdentity(int size);

/** @brief Construct an rectangular identity matrix (as a SparseMatrix)
    @param rows number of row
    @param cols number of columns
    @param row_offset offset row where diagonal identity starts
    @param col_offset offset column where diagonal identity starts
*/
template <typename T = double>
SparseMatrix<T> SparseIdentity(int rows, int cols, int row_offset = 0, int col_offset = 0);

/** @brief Adds two sparse matrices C = alpha * A + beta * B
           Nonzero entries do not have to match between A and B

    @param alpha scale for A
    @param A A matrix
    @param beta scale for B
    @param B B matrix
    @returns C such that C = alpha * A + beta * B
*/
template <typename T = double>
SparseMatrix<T> Add(double alpha, const SparseMatrix<T>& A,
                    double beta, const SparseMatrix<T>& B);

/** @brief Adds two sparse matrices C = A + B
           Nonzero entries do not have to match between A and B

    @param A A matrix
    @param B B matrix
    @returns C such that C = A + B
*/
template <typename T = double>
SparseMatrix<T> Add(const SparseMatrix<T>& A,
                    const SparseMatrix<T>& B);

/** @brief Change the type of sparse matrix through copy

    @param input Input matrix
    @param output Output matrix of desired output type
    @returns C such that C = A + B
*/
template <typename T, typename U>
SparseMatrix<U> Duplicate(const SparseMatrix<T>& input);

///
/// Inline implementations:
///

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

template <typename T>
SparseMatrix<T> SparseIdentity(int size)
{
    assert(size >= 0);

    return SparseMatrix<T>(std::vector<T>(size, (T)1.0));
}

template <typename T>
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

template <typename T>
SparseMatrix<T> Add(const SparseMatrix<T>& A,
                    const SparseMatrix<T>& B)
{
    return Add(1.0, A, 1.0, B);
}

template <typename T>
SparseMatrix<T> Add(double alpha, const SparseMatrix<T>& A,
                    double beta, const SparseMatrix<T>& B)
{
    assert(A.Rows() == B.Rows());
    assert(A.Cols() == B.Cols());

    CooMatrix<T> coo(A.Rows(), A.Cols());

    auto add_mat = [&coo](double scale, const SparseMatrix<T> & mat)
    {
        const auto& indptr = mat.GetIndptr();
        const auto& indices = mat.GetIndices();
        const auto& data = mat.GetData();

        int rows = mat.Rows();

        for (int i = 0; i < rows; ++i)
        {
            for (int j = indptr[i]; j < indptr[i + 1]; ++j)
            {
                coo.Add(i, indices[j], scale * data[j]);
            }
        }
    };

    add_mat(alpha, A);
    add_mat(beta, B);

    return coo.ToSparse();
}

template <typename T, typename U>
SparseMatrix<U> Duplicate(const SparseMatrix<T>& input)
{
    auto indptr = input.GetIndptr();
    auto indices = input.GetIndices();
    std::vector<U> data(input.GetData().size());

    const auto& input_data = input.GetData();
    std::copy(std::begin(input_data), std::end(input_data), std::begin(data));

    return SparseMatrix<U>(std::move(indptr), std::move(indices), std::move(data),
                           input.Rows(), input.Cols());
}

} //namespace linalgcpp

#endif // UTILITIES_HPP__

