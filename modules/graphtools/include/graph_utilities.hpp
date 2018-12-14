#ifndef GRAPH_UTILITIES_HPP
#define GRAPH_UTILITIES_HPP

#include "partition.hpp"
#include "parlinalgcpp.hpp"

namespace linalgcpp
{

/** @brief Creates relationship entity_set

    @param partition partition of entities into sets
    @returns entity_set
*/
template <typename T = double>
SparseMatrix<T> MakeEntitySet(std::vector<int> partition);

/** @brief Creates relationship set_entity

    @param partition partition of entities into sets
    @returns set_entity
*/
template <typename T = double>
SparseMatrix<T> MakeSetEntity(std::vector<int> partition);

/** @brief Remove entries below the tolerance completely

    @param mat matrix from with to remove entries
    @param tol remove entries below this tolerance
    @returns matrix without entries below tolerance
*/
template <typename T = double>
SparseMatrix<T> RemoveLargeEntries(const SparseMatrix<T>& mat, double tol = 1.0);

/** @brief Remove entries below the tolerance completely

    @param mat matrix from with to remove entries
    @param tol remove entries below this tolerance
    @returns matrix without entries below tolerance
*/
ParMatrix RemoveLargeEntries(const ParMatrix& mat, double tol = 1.0);

/** @brief Partitions matrix = A * A^T

    @param A matrix to partition
    @param coarsening_factor determine number of parts to partition into
    @returns partitioning of A * A^T
*/
template <typename T = double>
std::vector<int> PartitionAAT(const SparseMatrix<T>& A, double coarsening_factor,
                              double ubal = 1.0, bool contig = true);

/** @brief Create entity to true entity relationship
    @param entity_entity entity to entity relationship on false dofs
    @return entity_true_entity entity to true entity
*/
ParMatrix MakeEntityTrueEntity(const ParMatrix& entity_entity);

/** @brief Read serial vector from file and extract local portion

    @param filename name of vector file
    @param local_to_global set of local indices to extract
    @returns local vector
*/
template <typename T=double>
Vector<T> ReadVector(const std::string& filename,
                     const std::vector<int>& local_to_global);

/// SparseMatrix triple product
template <typename T = double>
SparseMatrix<T> Mult(const SparseMatrix<T>& R, const SparseMatrix<T>& A, const SparseMatrix<T>& P);

/// ParMatrix triple product
ParMatrix Mult(const ParMatrix& R, const ParMatrix& A, const ParMatrix& P);

/// Nonzero density of a matrix
template <typename T = double>
double Density(const SparseMatrix<T>& A);

/** @brief Distribute sets to processors
*/
template <typename T = double>
SparseMatrix<T> MakeProcAgg(MPI_Comm comm, const SparseMatrix<T>& agg_vertex,
                         const SparseMatrix<T>& vertex_edge);

/** @brief Create an edge to true edge relationship
    @param comm MPI Communicator
    @param proc_edge processor edge relationship
    @param edge_map local to global edge map
    @returns global edge to true edge
*/
template <typename T = double>
ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix<T>& proc_edge,
                           const std::vector<int>& edge_map);

/// Shifts partition such that indices are in [0, num_parts]
template <typename T = int>
void ShiftPartition(std::vector<T>& partition);

template <typename T, typename U>
SparseMatrix<U> Duplicate(const SparseMatrix<T>& input);

ParMatrix MakeExtPermutation(const ParMatrix& parmat);

////////////////////////////////
// Templated Implementations  //
////////////////////////////////

template <typename T>
SparseMatrix<T> Mult(const SparseMatrix<T>& R, const SparseMatrix<T>& A, const SparseMatrix<T>& P)
{
    return R.Mult(A).Mult(P);
}

template <typename T>
SparseMatrix<T> MakeEntitySet(std::vector<int> partition)
{
    if (partition.size() == 0)
    {
        return SparseMatrix<T>();
    }

    const int num_parts = *std::max_element(std::begin(partition), std::end(partition)) + 1;
    const int num_vert = partition.size();

    std::vector<int> indptr(num_vert + 1);
    std::vector<T> data(num_vert, 1.0);

    std::iota(std::begin(indptr), std::end(indptr), 0);

    return SparseMatrix<T>(std::move(indptr), std::move(partition), std::move(data),
                        num_vert, num_parts);
}

template <typename T>
SparseMatrix<T> MakeSetEntity(std::vector<int> partition)
{
    return MakeEntitySet<T>(std::move(partition)).Transpose();
}

template <typename T>
SparseMatrix<T> RemoveLargeEntries(const SparseMatrix<T>& mat, double tol)
{
    int rows = mat.Rows();
    int cols = mat.Cols();

    const auto& mat_indptr = mat.GetIndptr();
    const auto& mat_indices = mat.GetIndices();
    const auto& mat_data = mat.GetData();

    std::vector<int> indptr(rows + 1);
    std::vector<int> indices;

    indices.reserve(mat.nnz());

    for (int i = 0; i < rows; ++i)
    {
        indptr[i] = indices.size();

        for (int j = mat_indptr[i]; j < mat_indptr[i + 1]; ++j)
        {
            if (std::fabs(mat_data[j]) > tol)
            {
                indices.push_back(mat_indices[j]);
            }
        }
    }

    indptr[rows] = indices.size();

    std::vector<T> data(indices.size(), 1);

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                           rows, cols);
}

template <typename T>
std::vector<int> PartitionAAT(const SparseMatrix<T>& A, double coarsening_factor,
                              double ubal, bool contig)
{
    SparseMatrix<T> A_T = A.Transpose();
    SparseMatrix<T> AA_T = A.Mult(A_T);

    int num_parts = std::max(1.0, (A.Rows() / coarsening_factor) + 0.5);

    return Partition(AA_T, num_parts, ubal, contig);
}

template <typename T>
Vector<T> ReadVector(const std::string& filename,
                     const std::vector<int>& local_to_global)
{
    std::vector<T> global_vect = linalgcpp::ReadText<T>(filename);

    int local_size = local_to_global.size();

    Vector<T> local_vect(local_size);

    for (int i = 0; i < local_size; ++i)
    {
        local_vect[i] = global_vect[local_to_global[i]];
    }

    return local_vect;
}

template <typename T>
double Density(const SparseMatrix<T>& A)
{

    double denom = A.Rows() * (double) A.Cols();
    return A.nnz() / denom;
}

template <typename T>
SparseMatrix<T> MakeProcAgg(MPI_Comm comm, const SparseMatrix<T>& agg_vertex,
                         const SparseMatrix<T>& vertex_edge)
{
    int num_procs;
    int num_aggs = agg_vertex.Rows();

    MPI_Comm_size(comm, &num_procs);

    if (num_procs == 0)
    {
        std::vector<int> trivial_partition(num_aggs, 0);
        return MakeSetEntity<T>(std::move(trivial_partition));
    }

    SparseMatrix<T> agg_edge = agg_vertex.Mult(vertex_edge);
    SparseMatrix<T> agg_agg = agg_edge.Mult(agg_edge.Transpose());

    // Metis doesn't behave well w/ very dense sparse partition
    // so we partition by hand if aggregates are densely connected
    const double density = Density(agg_agg);
    const double density_tol = 0.7;

    std::vector<int> partition;

    if (density < density_tol)
    {
        double ubal = 1.0;
        partition = Partition(agg_agg, num_procs, ubal);
    }
    else
    {
        partition.reserve(num_aggs);

        int num_each = num_aggs / num_procs;
        int num_left = num_aggs % num_procs;

        for (int proc = 0; proc < num_procs; ++proc)
        {
            int local_num = num_each + (proc < num_left ? 1 : 0);

            for (int i = 0; i < local_num; ++i)
            {
                partition.push_back(proc);
            }
        }

        assert(static_cast<int>(partition.size()) == num_aggs);
    }

    SparseMatrix<T> proc_agg = MakeSetEntity<T>(std::move(partition));

    assert(proc_agg.Cols() == num_aggs);
    assert(proc_agg.Rows() == num_procs);

    return proc_agg;
}

template <typename T>
ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix<T>& proc_edge,
                           const std::vector<int>& edge_map)
{
    int myid;
    int num_procs;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    SparseMatrix<T> edge_proc = proc_edge.Transpose();

    int num_edges_local = proc_edge.RowSize(myid);
    int num_tedges_global = proc_edge.Cols();

    std::vector<int> tedge_counter(num_procs + 1, 0);

    for (int i = 0; i < num_tedges_global; ++i)
    {
        tedge_counter[edge_proc.GetIndices(i)[0] + 1]++;
    }

    int num_tedges_local = tedge_counter[myid + 1];
    int num_edge_diff = num_edges_local - num_tedges_local;
    std::partial_sum(std::begin(tedge_counter), std::end(tedge_counter),
                     std::begin(tedge_counter));

    assert(tedge_counter.back() == static_cast<int>(num_tedges_global));

    std::vector<int> edge_perm(num_tedges_global);

    for (int i = 0; i < num_tedges_global; ++i)
    {
        edge_perm[i] = tedge_counter[edge_proc.GetIndices(i)[0]]++;
    }

    for (int i = num_procs - 1; i > 0; i--)
    {
        tedge_counter[i] = tedge_counter[i - 1];
    }
    tedge_counter[0] = 0;

    std::vector<int> diag_indptr(num_edges_local + 1);
    std::vector<int> diag_indices(num_tedges_local);
    std::vector<double> diag_data(num_tedges_local, 1.0);

    std::vector<int> offd_indptr(num_edges_local + 1);
    std::vector<int> offd_indices(num_edge_diff);
    std::vector<double> offd_data(num_edge_diff, 1.0);
    std::vector<HYPRE_Int> col_map(num_edge_diff);
    std::vector<std::pair<HYPRE_Int, int>> offd_map(num_edge_diff);

    diag_indptr[0] = 0;
    offd_indptr[0] = 0;

    int tedge_begin = tedge_counter[myid];
    int tedge_end = tedge_counter[myid + 1];

    int diag_counter = 0;
    int offd_counter = 0;

    for (int i = 0; i < num_edges_local; ++i)
    {
        int tedge = edge_perm[edge_map[i]];

        if ((tedge >= tedge_begin) && (tedge < tedge_end))
        {
            diag_indices[diag_counter++] = tedge - tedge_begin;
        }
        else
        {
            offd_map[offd_counter].first = tedge;
            offd_map[offd_counter].second = offd_counter;
            offd_counter++;
        }

        diag_indptr[i + 1] = diag_counter;
        offd_indptr[i + 1] = offd_counter;
    }

    assert(offd_counter == static_cast<int>(num_edge_diff));

    auto compare = [] (const std::pair<HYPRE_Int, int>& lhs,
                       const std::pair<HYPRE_Int, int>& rhs)
    {
        return lhs.first < rhs.first;
    };

    std::sort(std::begin(offd_map), std::end(offd_map), compare);

    for (int i = 0; i < offd_counter; ++i)
    {
        offd_indices[offd_map[i].second] = i;
        col_map[i] = offd_map[i].first;
    }

    auto starts = linalgcpp::GenerateOffsets(comm, {num_edges_local, num_tedges_local});

    SparseMatrix<double> diag(std::move(diag_indptr), std::move(diag_indices), std::move(diag_data),
                              num_edges_local, num_tedges_local);

    SparseMatrix<double> offd(std::move(offd_indptr), std::move(offd_indices), std::move(offd_data),
                      num_edges_local, num_edge_diff);

    return ParMatrix(comm, starts[0], starts[1],
                     std::move(diag), std::move(offd),
                     std::move(col_map));
}

template <typename T>
void ShiftPartition(std::vector<T>& partition)
{
    int min_part = *std::min_element(std::begin(partition), std::end(partition));

    for (auto& i : partition)
    {
        i -= min_part;
    }

    linalgcpp::RemoveEmpty(partition);
}


} // namespace linalgcpp

#endif // GRAPH_UTILITIES_HPP
