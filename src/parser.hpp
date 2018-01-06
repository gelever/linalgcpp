/*! @file */
#ifndef PARSER_HPP__
#define PARSER_HPP__

#include <memory>
#include <vector>
#include <assert.h>
#include <fstream>

#include "vector.hpp"
#include "sparsematrix.hpp"
#include "coomatrix.hpp"

namespace linalgcpp
{

/*! @brief Read a text file from disk.
    @param file_name file to read
    @retval vector of data read from disk
*/
template <typename T = double>
std::vector<T> ReadText(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    std::vector<T> data;

    for (T val; file >> val; )
    {
        data.push_back(val);
    }

    file.close();

    return data;
}

/*! @brief Write a vector to a text file on disk.
    @param vect vector to write
    @param file_name file to write to
*/
template <typename T = double>
void WriteText(const std::vector<T>& vect, const std::string& file_name)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    file.precision(16);

    for (const T& val : vect)
    {
        file << val << "\n";
    }
}

/*! @brief Read an adjacency list from disk.
    Data is expected to be formatted as :
       i j
       i j
       i j
       ...
    @param file_name file to read
    @param symmetric if true the file only contain values above
    or below the diagonal and the diagonal itself. The other corresponding
    symmetric values will be added to the matrix.
*/
template <typename T = double>
SparseMatrix<T> ReadAdjList(const std::string& file_name, bool symmetric = false)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    CooMatrix<T> coo;

    int i;
    int j;
    T val = 1;

    while (file >> i >> j)
    {
        coo.Add(i, j, val);

        if (symmetric && i != j)
        {
            coo.Add(j, i, val);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Read graph laplacian from an adjacency list from disk.
    Data is expected to be formatted as :
       i j
       i j
       i j
       ...
    @param file_name file to read
*/
template <typename T = double>
SparseMatrix<T> ReadGraphList(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    CooMatrix<T> coo;

    int i;
    int j;

    while (file >> i >> j)
    {
        if (i < j)
        {
            coo.Add(j, i, -1);
            coo.Add(i, j, -1);
            coo.Add(i, i, 1);
            coo.Add(j, j, 1);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Write an adjacency list to disk.
    @param mat matrix to write out
    @param file_name file to write to
    @param symmetric if true only write entries above and including the diagonal.
    Otherwise write out all entries

    @note see ReadAdjList for format description
*/
template <typename T = double>
void WriteAdjList(const SparseMatrix<T>& mat, const std::string& file_name, bool symmetric = false)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const int col = indices[j];

            if (!symmetric || col >= i)
            {
                file << i << " " << col << "\n";
            }
        }
    }

    file.close();
}

/*! @brief Read a coordinate list from disk.
    Data is expected to be formatted as :
       i j val
       i j val
       i j val
       ...
    @param file_name file to read
    @param symmetric if true the file only contain values above
    or below the diagonal and the diagonal itself. The other corresponding
    symmetric values will be added to the matrix.
*/
template <typename T = double>
SparseMatrix<T> ReadCooList(const std::string& file_name, bool symmetric = false)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    CooMatrix<T> coo;

    int i;
    int j;
    T val;

    while (file >> i >> j >> val)
    {
        coo.Add(i, j, val);

        if (symmetric && i != j)
        {
            coo.Add(j, i, val);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Write a coordinate list to disk.
    @param mat matrix to write out
    @param file_name file to write to
    @param symmetric if true only write entries above and including the diagonal.
    Otherwise write out all entries

    @note see ReadCooList for format description
*/
template <typename T = double>
void WriteCooList(const SparseMatrix<T>& mat, const std::string& file_name, bool symmetric = false)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();
    const std::vector<double>& data = mat.GetData();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const int col = indices[j];

            if (!symmetric || col >= i)
            {
                file << i << " " << col << " " << data[j] << "\n";
            }
        }
    }

    file.close();
}

/*! @brief Read a table from file from disk.
    Data is expected to be formatted as :
        0 1 2 3 4
        5 8 0
        1 2 3 10
    Where each row corresponds to the entries in that row
    @param file_name file to read
    @retval SparseMatrix matrix of data read from disk
*/
template <typename T = int>
SparseMatrix<T> ReadTable(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    std::vector<int> indptr(1, 0);
    std::vector<int> indices;

    for (std::string line; std::getline(file, line); )
    {
        std::stringstream stream(line);

        for (int index; stream >> index; )
        {
            indices.push_back(index);
        }

        indptr.push_back(indices.size());
    }

    file.close();

    const int rows = indptr.size() - 1;
    const int cols = *std::max_element(begin(indices), end(indices)) + 1;

    std::vector<T> data(indices.size(), 1);

    return SparseMatrix<T>(indptr, indices, data,
                           rows, cols);
}

/*! @brief Write a table to a file on disk.
    @param mat table to write
    @param file_name file to write to

    @note see ReadTable for format description
*/
template <typename T = int>
void WriteTable(const SparseMatrix<T>& mat, const std::string& file_name)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const std::string space = j + 1 == indptr[i + 1] ? "" : " ";
            file << indices[j] << space;
        }

        file << "\n";
    }

    file.close();
}

} // namespace mylinalg

#endif // PARSER_HPP__
