#include "graph_mis.hpp"

namespace linalgcpp
{

ParMatrix SelectMIS(const ParMatrix& mis_dof)
{
    const auto& mis_dof_diag = mis_dof.GetDiag();

    int num_mis = mis_dof.Rows();

    std::vector<int> indptr(1, 0);
    std::vector<int> indices;

    for (int i = 0; i < num_mis; ++i)
    {
        if (mis_dof_diag.RowSize(i) > 0)
        {
            indptr.push_back(indptr.size());
            indices.push_back(i);
        }
    }

    std::vector<double> data(indices.size(), 1.0);

    int num_selected = indices.size();

    SparseMatrix<double> selector(std::move(indptr), std::move(indices), std::move(data),
                                  num_selected, num_mis);

    ParMatrix selector_par(mis_dof.GetComm(), std::move(selector));

    return RemoveLargeEntries(selector_par.Mult(mis_dof), 0.0);
}

ParMatrix MakeMISDof(const ParMatrix& agg_dof)
{
    // Redistribute neighbor aggregate info
    ParMatrix dof_agg = agg_dof.Transpose();
    ParMatrix agg_agg = agg_dof.Mult(dof_agg);
    ParMatrix dof_dof = dof_agg.Mult(agg_dof);

    ParMatrix agg_r = MakeExtPermutation(agg_agg);
    ParMatrix dof_r = MakeExtPermutation(dof_dof);
    ParMatrix dof_r_T = dof_r.Transpose();

    ParMatrix agg_dof_r = agg_r.Mult(agg_dof).Mult(dof_r_T);

    SparseMatrix<double> agg_dof_ext = agg_dof_r.GetDiag();
    agg_dof_ext.EliminateZeros();

    SparseMatrix<double> dof_agg_ext = agg_dof_ext.Transpose();

    // Run serial MIS algorithm on redistributed, local agg_dof
    auto mis = GenerateMIS(agg_dof_ext, dof_agg_ext);
    SparseMatrix<double> mis_dof = MakeSetEntity<double>(std::move(mis));

    // Obtain true_mis, since local mis may now be duplicated
    ParMatrix mis_dof_par(agg_dof.GetComm(), std::move(mis_dof));
    ParMatrix mis_dof_r = mis_dof_par.Mult(dof_r);
    ParMatrix true_mis_true_dof = SelectMIS(mis_dof_r);

    return RemoveLargeEntries(true_mis_true_dof, 0.0);
}


} // namespace linalgcpp
