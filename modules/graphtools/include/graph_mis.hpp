
/** @file

    @brief Implementations of distributed MIS algorithm
*/

#ifndef MIS_HPP
#define MIS_HPP

#include "graph.hpp"
#include "graph_topology.hpp"
#include "graph_utilities.hpp"


namespace linalgcpp
{

template <typename T = double>
std::vector<int> GenerateMIS(const SparseMatrix<T>& set_dof,
                             const SparseMatrix<T>& dof_set);

ParMatrix SelectMIS(const ParMatrix& mis_dof);

template <typename T, typename U, typename V>
ParMatrix MakeMISDof(const Graph<T, U, V>& graph, const GraphTopology<T>& topo);

ParMatrix MakeMISDof(const ParMatrix& agg_dof);

//SparseMatrix MakeFaceMIS(const SparseMatrix& mis_agg);


///////////////////////
/// Implementations ///
///////////////////////

template <typename T>
std::vector<int> GenerateMIS(const SparseMatrix<T>& set_dof,
                             const SparseMatrix<T>& dof_set)
{
    const auto& set_dof_I = set_dof.GetIndptr();
    const auto& set_dof_J = set_dof.GetIndices();

    const auto& dof_set_I = dof_set.GetIndptr();
    const auto& dof_set_J = dof_set.GetIndices();

    int num_dofs = dof_set.Rows();
    int num_mises = 0;

    std::vector<int> distributed(num_dofs, false);
    std::vector<int> mises(num_dofs, 0);

    // Loop over all DoFs and build MISes.
    for (int i = 0; i < num_dofs; ++i)
    {
        // If DoF i is already assigned.
        if (distributed[i])
        {
            continue;
        }

        // Loop over all AEs that contain DoF i.
        for (int j = dof_set_I[i]; j < dof_set_I[i+1]; ++j)
        {
            // Loop over all DoFs in these AEs.
            for (int k = set_dof_I[dof_set_J[j]];
                 k < set_dof_I[dof_set_J[j]+1]; ++k)
            {
                if (!distributed[set_dof_J[k]])
                {
                    ++mises[set_dof_J[k]]; // Count the current AE.
                }
            }
        }

        // Loop over all AEs that contain DoF i.
        for (int j = dof_set_I[i]; j < dof_set_I[i+1]; ++j)
        {
            // Loop over all DoFs in these AEs.
            for (int k = set_dof_I[dof_set_J[j]];
                 k < set_dof_I[dof_set_J[j]+1]; ++k)
            {
                if (distributed[set_dof_J[k]])
                {
                    continue;
                }
                // If the DoF should be in the same MIS as DoF i.
                if (mises[set_dof_J[k]] == dof_set_I[i+1] -
                                             dof_set_I[i] &&
                    mises[set_dof_J[k]] == dof_set_I[set_dof_J[k]+1] -
                                             dof_set_I[set_dof_J[k]])
                {
                    mises[set_dof_J[k]] = num_mises;
                    distributed[set_dof_J[k]] = true;
                } else
                {
                    mises[set_dof_J[k]] = 0;
                }
            }
        }

        ++num_mises;
    }

    return mises;
}

template <typename T, typename U, typename V>
ParMatrix MakeMISDof(const Graph<T, U, V>& graph, const GraphTopology<T>& topo)
{
    MPI_Comm comm = graph.edge_true_edge_.GetComm();

    auto agg_edge_local = topo.agg_vertex_local_.template Mult<T, double>(graph.vertex_edge_local_);
    ParMatrix agg_edge_par(comm, std::move(agg_edge_local));

    ParMatrix agg_edge = agg_edge_par.Mult(topo.edge_true_edge_);

    return MakeMISDof(agg_edge);
}

} //namespace linalgcpp

#endif // MIS_HPP
