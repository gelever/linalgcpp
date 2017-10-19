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

template <typename T = double>
std::vector<T> ReadText(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file!\n");
    }

    std::vector<T> data;

    for (T val; file >> val; )
    {
        data.push_back(val);
    }

    return data;
}

template <typename T = double>
SparseMatrix<T> ReadAdjList(const std::string& file_name, bool symmetric = false)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file!\n");
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

    return coo.ToSparse();
}

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

    return coo.ToSparse();
}
        



}

#endif // PARSER_HPP__
