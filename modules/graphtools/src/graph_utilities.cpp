#include "graph_utilities.hpp"

namespace linalgcpp
{

ParMatrix RemoveLargeEntries(const ParMatrix& mat, double tol)
{
    int num_rows = mat.Rows();

    const auto& diag_ext = mat.GetDiag();
    const auto& offd_ext = mat.GetOffd();
    const auto& colmap_ext = mat.GetColMap();

    const auto& offd_indptr = offd_ext.GetIndptr();
    const auto& offd_indices = offd_ext.GetIndices();
    const auto& offd_data = offd_ext.GetData();
    const int num_offd = offd_ext.Cols();

    std::vector<int> indptr(num_rows + 1);
    std::vector<int> offd_marker(num_offd, -1);

    int offd_nnz = 0;

    for (int i = 0; i < num_rows; ++i)
    {
        indptr[i] = offd_nnz;

        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            if (std::fabs(offd_data[j]) > tol)
            {
                offd_marker[offd_indices[j]] = 1;
                offd_nnz++;
            }
        }
    }

    indptr[num_rows] = offd_nnz;

    int offd_num_cols = std::count_if(std::begin(offd_marker), std::end(offd_marker),
    [](int i) { return i > 0; });

    std::vector<HYPRE_Int> col_map(offd_num_cols);
    int count = 0;

    for (int i = 0; i < num_offd; ++i)
    {
        if (offd_marker[i] > 0)
        {
            offd_marker[i] = count;
            col_map[count] = colmap_ext[i];

            count++;
        }
    }

    assert(count == offd_num_cols);

    std::vector<int> indices(offd_nnz);
    std::vector<double> data(offd_nnz, 1.0);

    count = 0;

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            if (offd_data[j] > 1)
            {
                indices[count++] = offd_marker[offd_indices[j]];
            }
        }
    }

    assert(count == offd_nnz);

    SparseMatrix<double> diag = RemoveLargeEntries(diag_ext);
    SparseMatrix<double> offd(std::move(indptr), std::move(indices), std::move(data),
                      num_rows, offd_num_cols);

    return ParMatrix(mat.GetComm(), mat.GetRowStarts(), mat.GetColStarts(),
                     std::move(diag), std::move(offd), std::move(col_map));
}

ParMatrix MakeEntityTrueEntity(const ParMatrix& entity_entity)
{
    const auto& offd = entity_entity.GetOffd();

    const auto& offd_indptr = offd.GetIndptr();
    const auto& offd_indices = offd.GetIndices();
    const auto& offd_colmap = entity_entity.GetColMap();

    HYPRE_Int last_row = entity_entity.GetColStarts()[1];

    int num_entities = entity_entity.Rows();
    std::vector<int> select_indptr(num_entities + 1);

    int num_true_entities = 0;

    for (int i = 0; i < num_entities; ++i)
    {
        select_indptr[i] = num_true_entities;

        int row_size = offd.RowSize(i);

        if (row_size == 0)
        {
            num_true_entities++;
        }
        else
        {
            bool owner = true;

            int start = offd_indptr[i];
            int end = offd_indptr[i + 1];

            for (int j = start; j < end; ++j)
            {
                if (offd_colmap[offd_indices[j]] < last_row)
                {
                    owner = false;
                    break;
                }
            }

            if (owner)
            {
                num_true_entities++;
            }
        }
    }

    select_indptr[num_entities] = num_true_entities;

    std::vector<int> select_indices(num_true_entities);
    std::iota(std::begin(select_indices), std::end(select_indices), 0);

    std::vector<double> select_data(num_true_entities, 1.0);

    SparseMatrix<double> select(std::move(select_indptr), std::move(select_indices), std::move(select_data),
                        num_entities, num_true_entities);

    MPI_Comm comm = entity_entity.GetComm();
    auto true_starts = linalgcpp::GenerateOffsets(comm, num_true_entities);

    ParMatrix select_d(comm, entity_entity.GetRowStarts(), true_starts, std::move(select));

    return entity_entity.Mult(select_d);
}


ParMatrix Mult(const ParMatrix& R, const ParMatrix& A, const ParMatrix& P)
{
    return R.Mult(A).Mult(P);
}

ParMatrix MakeExtPermutation(const ParMatrix& parmat)
{
    MPI_Comm comm = parmat.GetComm();

    const auto& diag = parmat.GetDiag();
    const auto& offd = parmat.GetOffd();
    const auto& colmap = parmat.GetColMap();

    int num_diag = diag.Cols();
    int num_offd = offd.Cols();
    int num_ext = num_diag + num_offd;

    const auto& mat_starts = parmat.GetColStarts();
    auto ext_starts = linalgcpp::GenerateOffsets(comm, num_ext);

    SparseMatrix<double> perm_diag = SparseIdentity(num_ext, num_diag);
    SparseMatrix<double> perm_offd = SparseIdentity(num_ext, num_offd, num_diag);

    return ParMatrix(comm, ext_starts, mat_starts,
                     std::move(perm_diag), std::move(perm_offd),
                     colmap);
}



} // namespace linalgcpp
