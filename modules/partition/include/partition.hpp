/*! @file */

#ifndef PARTITION_HPP__
#define PARTITION_HPP__

#include <tuple>

#include "metis.h"
#include "linalgcpp.hpp"

namespace linalgcpp
{

template <typename T, typename U>
std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
MakeAdjacencyInfo(const linalgcpp::SparseMatrix<T>& mat, bool weighted = false)
{
    assert(mat.Cols() == mat.Rows());

    int num_edges = (mat.nnz() - mat.Rows()) / 2;

    std::vector<U> indptr(mat.Rows() + 1);
    std::vector<U> indices;
    std::vector<U> data;

    indices.reserve(num_edges);
    data.reserve(num_edges);

    indptr[0] = 0;

    for (int i = 0; i < mat.Rows(); ++i)
    {
        for (int j = mat.GetIndptr()[i]; j < mat.GetIndptr()[i + 1]; ++j)
        {
            const int col = mat.GetIndices()[j];

            if (col != i)
            {
                indices.push_back(col);
                data.push_back(weighted ? std::fabs(mat.GetData()[j]) : 1.0);
            }
        }

        indptr[i + 1] = indices.size();
    }

    return std::make_tuple(std::move(indptr), std::move(indices), std::move(data));
}

inline
void RemoveEmpty(std::vector<int>& partition)
{
    assert(partition.size() > 0);

    const int num_parts = *std::max_element(std::begin(partition), std::end(partition)) + 1;

    std::vector<int> counter(num_parts, 0);
    std::vector<int> shifter(num_parts);

    for (const auto& i : partition)
    {
        ++counter[i];
    }

    int shift = 0;

    for (int i = 0; i < num_parts; ++i)
    {
        if (counter[i] == 0)
        {
            ++shift;
        }

        shifter[i] = shift;
    }

    const int num_vert = partition.size();

    for (int i = 0; i < num_vert; ++i)
    {
        partition[i] -= shifter[partition[i]];
    }
}

template <typename T, typename U>
std::vector<U> Duplicate(const std::vector<T>& vect)
{
    std::vector<U> vect_copy(vect.size());
    std::copy(std::begin(vect), std::end(vect), std::begin(vect_copy));

    return vect_copy;
}

/** @brief Wrapper to call Metis Partitioning
    @param mat graph to partition
    @param num_parts number of partitions to generate
    @param unbalance_factor allows some unbalance in partition sizes,
           where 1.0 is little unbalance and 2.0 is lots of unbalance
    @param contig generate only contiguous partitions where the partitioned subgraphs are always connected,
            requires the input graph be connected
    @param weighted use the input graph values as edge weights.
    @warning  Metis requires positive integer edge weights, so the absolute value is taken and converted to integer.
           Scale the input appropriately to obtain desired weights
    @retval partition vector with values representing the partition of the index
 */
template <typename T>
std::vector<int> Partition(const linalgcpp::SparseMatrix<T>& mat, int num_parts = 2,
                           double unbalance_factor = 1.0, bool contig = true, bool weighted = false)
{
    assert(num_parts > 0);
    assert(num_parts <= mat.Rows());
    assert(mat.Cols() == mat.Rows());

    const int size = mat.Rows();

    if (num_parts < 2)
    {
        return std::vector<int>(size, 0);
    }

    std::vector<idx_t> part(size);

    auto adj_pair = MakeAdjacencyInfo<T, idx_t>(mat, weighted);
    idx_t* xadj = std::get<0>(adj_pair).data();
    idx_t* adjncy = std::get<1>(adj_pair).data();
    idx_t* adjwgt = std::get<2>(adj_pair).data();

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = contig;
    options[METIS_OPTION_NUMBERING] = 0;

    idx_t nvtxs = size;
    idx_t ncon = 1;

    idx_t* vwgt = nullptr;
    idx_t* vsize = nullptr;
    idx_t nparts = num_parts;
    real_t* tpwgts = nullptr;
    real_t ubvec = unbalance_factor;

    idx_t objval;

    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                             &nparts, tpwgts, &ubvec, options, &objval, part.data());

    RemoveEmpty(part);

    auto partition = std::is_same<idx_t, int>::value ? std::move(part) : Duplicate<idx_t, int>(part);

    return partition;
}

} // namespace linalgcpp

#endif // PARTITION_HPP__
