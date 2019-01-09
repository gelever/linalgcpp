#ifndef PARPARTITION_HPP
#define PARPARTITION_HPP

#include "graph_utilities.hpp"

namespace linalgcpp
{

ParMatrix ParPartition(const ParMatrix& A, int num_parts);
ParMatrix UnDistributer(const ParMatrix& A);

void ParNormalizeRows(ParMatrix& A);

void SortColumnMap(std::vector<int>& col_map, std::vector<int>& indices);

double ComputeT(const linalgcpp::SparseMatrix<double>& A);
double ComputeT(const ParMatrix& A);

double CalcQ(const linalgcpp::SparseMatrix<double>& A,
             const linalgcpp::SparseMatrix<double>& P);
double CalcQ(const ParMatrix& A, const ParMatrix& PT);

} // namespace linalgcpp

#endif // PARPARTITION_HPP


