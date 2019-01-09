
/** @file

    @brief Implementations of distributed MIS algorithm
*/

#ifndef MIS_HPP
#define MIS_HPP

#include "graph.hpp"
#include "graph_utilities.hpp"


namespace linalgcpp
{


/** @brief Generate a partition of dof using minimum intersection algorithm.
 *         Serial implementation.
 *
 *  @param set_dof set to dof relationship
 *  @param dof_set dof to set relationship
 *  @returns partition vector indicating which MIS each dof belongs to
 */
template <typename T = double>
std::vector<int> GenerateMIS(const SparseMatrix<T>& set_dof,
                             const SparseMatrix<T>& dof_set);

/** @brief Select MISs that contain at least one dof on the current processor.
 *
 *  @param mis_dof distributed mis to dof relationship
 *  @returns mis_dof with off-processor MIS removed
 */
ParMatrix SelectMIS(const ParMatrix& mis_dof);

/** @brief Generate a MIS to dof relationship
 *
 *  @param agg_dof agglomerate to dof relationship
 *  @returns mis_dof mis to dof relationship
 */
ParMatrix MakeMISDof(const ParMatrix& agg_dof);

/** @brief Generate local face to mis relationship,
 *         where a face is any mis which belongs to more than
 *         one aggregate
 *
 *  @param mis_agg mis to aggregate relationship
 *  @returns face_mis face to mis relationship
 */
template <typename T = double>
SparseMatrix<T> MakeFaceMIS(const SparseMatrix<T>& mis_agg);

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

template <typename T>
SparseMatrix<T> MakeFaceMIS(const SparseMatrix<T>& mis_agg)
{
    std::vector<int> indptr(1, 0);
    std::vector<int> indices;

    int num_mis = mis_agg.Rows();

    for (int i = 0; i < num_mis; ++i)
    {
        if (mis_agg.RowSize(i) > 1)
        {
            indptr.push_back(indptr.size());
            indices.push_back(i);
        }
    }

    int num_faces = indices.size();
    std::vector<double> data(num_faces, 1.0);

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_mis);
}

template <typename T>
SparseMatrix<T> MakeFaceMIS(const ParMatrix& mis_agg)
{
    std::vector<int> indptr(1, 0);
    std::vector<int> indices;

    int num_mis = mis_agg.Rows();

    for (int i = 0; i < num_mis; ++i)
    {
        if (mis_agg.RowSize(i) > 1)
        {
            indptr.push_back(indptr.size());
            indices.push_back(i);
        }
    }

    int num_faces = indices.size();
    std::vector<double> data(num_faces, 1.0);

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_mis);
}

} //namespace linalgcpp

#endif // MIS_HPP
