/*! @file */

#include "parutilities.hpp"

namespace parlinalgcpp
{

ParMatrix ParSplit(MPI_Comm comm, const linalgcpp::SparseMatrix<double>& A_global,
                   const std::vector<int>& proc_part)
{
    assert(A_global.Rows() == A_global.Cols());

    int myid;
    int num_procs;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    linalgcpp::CooMatrix<double> local_diag;
    linalgcpp::CooMatrix<double> local_offd;
    std::vector<HYPRE_Int> col_map;

    std::vector<int> local_part;

    const int proc_size = proc_part.size();

    for (int i = 0; i < proc_size; ++i)
    {
        if (proc_part[i] == myid)
        {
            local_part.push_back(i);
        }
    }

    const int global_vertices = A_global.Rows();
    const int local_vertices = local_part.size();

    std::vector<HYPRE_Int> vertex_starts = GenerateOffsets(comm, local_part.size());

    std::vector<int> tmp(global_vertices, 0);

    for (int i = 0; i < local_vertices; ++i)
    {
        tmp[local_part[i]] = i + vertex_starts[0];
    }

    std::vector<int> perm(global_vertices, 0);
    MPI_Scan(tmp.data(), perm.data(), global_vertices, MPI::INT, MPI::SUM, comm);
    MPI_Bcast(perm.data(), global_vertices, MPI::INT, num_procs - 1, comm);
    const auto& indptr = A_global.GetIndptr();
    const auto& indices = A_global.GetIndices();
    const auto& data = A_global.GetData();

    std::vector<int> col_map_index(global_vertices, -1);
    std::vector<int> local_index(global_vertices, -1);

    for (int i = 0; i < local_vertices; ++i)
    {
        local_index[perm[local_part[i]]] = i;
    }

    for (int local_vertex = 0; local_vertex < local_vertices; ++local_vertex)
    {
        const int vertex = local_part[local_vertex];

        for (int j = indptr[vertex]; j < indptr[vertex + 1]; ++j)
        {
            int col = indices[j];
            double val = data[j];

            //if (std::fabs(val) < 1e-8)
            //{
            //    continue;
            //}

            if (perm[col] >= vertex_starts[0] && perm[col] < vertex_starts[1])
            {
                int local_col = local_index[perm[col]];
                assert(local_col >= 0 && local_col < local_vertices);

                if (local_col >= local_vertex)
                {
                    local_diag.AddSym(local_vertex, local_col, val);
                }
            }
            else
            {
                int local_col = col_map_index[perm[col]];

                if (local_col >= 0)
                {
                    local_offd.Add(local_vertex, local_col, val);
                }
                else
                {
                    col_map_index[perm[col]] = col_map.size();

                    local_offd.Add(local_vertex, col_map.size(), val);
                    col_map.push_back(perm[col]);
                }
            }
        }
    }

    local_diag.SetSize(local_vertices, local_vertices);
    local_offd.SetSize(local_vertices, col_map.size());
    SortColumnMap(col_map, local_offd);

    linalgcpp::SparseMatrix<double> diag = local_diag.ToSparse();
    linalgcpp::SparseMatrix<double> offd = local_offd.ToSparse();

    return ParMatrix(comm, vertex_starts, vertex_starts,
                     std::move(diag), std::move(offd), std::move(col_map));
}

std::vector<std::vector<HYPRE_Int>> GenerateOffsets(MPI_Comm comm, const std::vector<int>& local_sizes)
{
    assert(HYPRE_AssumedPartitionCheck());

    const int num_sizes = local_sizes.size();

    std::vector<std::vector<HYPRE_Int>> offsets(num_sizes, std::vector<HYPRE_Int>(3, 0));

    std::vector<int> scan(num_sizes, 0);

    MPI_Scan(local_sizes.data(), scan.data(), num_sizes,
             MPI_INT, MPI_SUM, comm);

    for (int i = 0; i < num_sizes; ++i)
    {
        offsets[i][0] = scan[i] - local_sizes[i];
        offsets[i][1] = scan[i];
    }

    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    MPI_Bcast(scan.data(), num_sizes, MPI_INT, num_procs - 1, comm);

    for (int i = 0; i < num_sizes; ++i)
    {
        offsets[i][2] = scan[i];
    }

    return offsets;
}

std::vector<HYPRE_Int> GenerateOffsets(MPI_Comm comm, int local_size)
{
    std::vector<int> local_sizes{local_size};
    std::vector<std::vector<HYPRE_Int>> offset = GenerateOffsets(comm, local_sizes);

    assert(offset.size() == 1);

    return offset[0];
}

void SortColumnMap(std::vector<HYPRE_Int>& col_map, linalgcpp::CooMatrix<double>& coo_offd)
{
    const int col_map_size = col_map.size();

    std::vector<int> permutation(col_map_size);
    std::iota(std::begin(permutation), std::end(permutation), 0);

    auto compare = [&](int i, int j)
    {
        return col_map[i] < col_map[j];
    };
    std::sort(std::begin(permutation), std::end(permutation), compare);

    std::vector<int> col_index(col_map_size);

    for (int i = 0; i < col_map_size; ++i)
    {
        col_index[permutation[i]] = i;
    }

    coo_offd.PermuteCols(col_index);

    for (int i = 0; i < col_map_size; ++i)
    {
        col_index[i] = col_map[permutation[i]];
    }

    std::swap(col_map, col_index);
}

ParMatrix Mult(const ParMatrix& lhs, const ParMatrix& rhs)
{
    return lhs.Mult(rhs);
}

ParMatrix Transpose(const ParMatrix& mat)
{
    return mat.Transpose();
}

ParMatrix RAP(const ParMatrix& R, const ParMatrix& A, const ParMatrix& P)
{
    ParMatrix R_T = R.Transpose();
    ParMatrix rap = R_T.Mult(A.Mult(P));

    return rap;
}

ParMatrix RAP(const ParMatrix& A, const ParMatrix& P)
{
    return RAP(P, A, P);
}

ParMatrix ParAdd(const ParMatrix& A, const ParMatrix& B)
{
    return ParAdd(1.0, A, 1.0, B);
}

ParMatrix ParAdd(double alpha, const ParMatrix& A, double beta, const ParMatrix& B)
{
    std::vector<int> marker(A.GlobalCols(), -1);
    return ParAdd(alpha, A, beta, B, marker);
}

ParMatrix ParAdd(double alpha, const ParMatrix& A, double beta, const ParMatrix& B,
                 std::vector<int>& marker)
{
    assert(A.Rows() == B.Rows());
    assert(A.Cols() == B.Cols());
    assert(A.GlobalRows() == B.GlobalRows());
    assert(A.GlobalCols() == B.GlobalCols());
    assert(A.GetRowStarts() == B.GetRowStarts());
    assert(A.GetComm() == B.GetComm());
    assert(static_cast<int>(marker.size()) >= A.GlobalCols());

    int rows = A.Rows();
    int cols = A.Cols();

    linalgcpp::CooMatrix<double> coo_diag(rows, cols);
    linalgcpp::CooMatrix<double> coo_offd;
    std::vector<HYPRE_Int> col_map;

    auto diag_to_coo = [&] (double val, const linalgcpp::SparseMatrix<double>& mat)
    {
        const auto& indptr = mat.GetIndptr();
        const auto& indices = mat.GetIndices();
        const auto& data = mat.GetData();

        for (int i = 0; i < rows; ++i)
        {
            for (int j = indptr[i]; j < indptr[i + 1]; ++j)
            {
                coo_diag.Add(i, indices[j], val * data[j]);
            }
        }
    };

    diag_to_coo(alpha, A.GetDiag());
    diag_to_coo(beta, B.GetDiag());

    // TODO(gelever): should this be assumed instead?
    std::fill(std::begin(marker), std::end(marker), -1);

    auto offd_to_coo = [&] (double val, const linalgcpp::SparseMatrix<double>& mat,
                            const std::vector<HYPRE_Int>& mat_col_map)
    {
        const auto& indptr = mat.GetIndptr();
        const auto& indices = mat.GetIndices();
        const auto& data = mat.GetData();

        for (int i = 0; i < rows; ++i)
        {
            for (int j = indptr[i]; j < indptr[i + 1]; ++j)
            {
                int global_col = mat_col_map[indices[j]];

                if (marker[global_col] >= 0)
                {
                    coo_offd.Add(i, marker[global_col], val * data[j]);
                }
                else
                {
                    coo_offd.Add(i, col_map.size(), data[j]);
                    marker[global_col] = col_map.size();

                    col_map.emplace_back(global_col);
                }
            }
        }
    };

    offd_to_coo(alpha, A.GetOffd(), A.GetColMap());
    offd_to_coo(beta, B.GetOffd(), B.GetColMap());

    coo_offd.SetSize(rows, col_map.size());
    SortColumnMap(col_map, coo_offd);

    for (auto& col : col_map)
    {
        marker[col] = -1;
    }

    auto diag = coo_diag.ToSparse();
    auto offd = coo_offd.ToSparse();

    return ParMatrix(A.GetComm(), A.GetRowStarts(), A.GetColStarts(),
                     std::move(diag), std::move(offd), std::move(col_map));
}

ParMatrix ParSub(const ParMatrix& A, const ParMatrix& B)
{
    return ParAdd(1.0, A, -1.0, B);
}

ParMatrix ParSub(double alpha, const ParMatrix& A, double beta, const ParMatrix& B)
{
    return ParAdd(alpha, A, -beta, B);
}

ParMatrix ParSub(double alpha, const ParMatrix& A, double beta, const ParMatrix& B,
                 std::vector<int>& marker)
{
    return ParAdd(alpha, A, -beta, B, marker);
}

} //namsespace parlinaglcpp
