/*! @file*/

#include <stdio.h>
#include <assert.h>

#include "linalgcpp.hpp"
#include "partition.hpp"

using namespace linalgcpp;

int main()
{
    CooMatrix<int> coo;

    const int size = 1e3;
    const int neighbors = 100;

    assert(size > 2 * neighbors);

    for (int i = 0; i < size; ++i)
    {
        coo.AddSym(i, i, 2 * neighbors);

        for (int j = i + 1; j < i + neighbors + 1; ++j)
        {
            coo.AddSym(i, j % size, -1);
        }
    }

    SparseMatrix<int> sparse = coo.ToSparse();

    std::cout << "nnz: " << sparse.nnz() << "\n";
    //sparse.PrintDense("Sparse:");

    const int coarse_factor = 100;
    const int num_parts = std::max(1, size / coarse_factor);
    const bool contig = true;
    const double unbalance_factor = 1.5;

    std::vector<int> part_default = Partition(sparse, num_parts);
    std::vector<int> part_fancy = Partition(sparse, num_parts, unbalance_factor, contig);
    std::vector<int> part_weighted = Partition(sparse, num_parts, unbalance_factor, contig, true);

    //std::cout << "Partition: " << part_default;
    //std::cout << "Partition: " << part_fancy;
}
