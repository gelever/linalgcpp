/*! @file */

#include "parprecond.hpp"

namespace linalgcpp
{

ParBlockDiagComp::ParBlockDiagComp(const ParMatrix& A, const ParMatrix& agg_vertex)
    : ParOperator(A), redist_(MakeRedistributor(agg_vertex)),
      sol_r_(redist_.Rows()), rhs_r_(redist_.Rows())
{
    ParMatrix redist_T = redist_.Transpose();
    ParMatrix agg_vertex_r = agg_vertex.Mult(redist_T);
    ParMatrix vertex_agg_r = agg_vertex_r.Transpose();
    ParMatrix A_r = redist_.Mult(A).Mult(redist_T);

    linalgcpp_verify(agg_vertex_r.GetOffd().nnz() == 0);

    block_op_ = BlockDiagOperator(agg_vertex_r.GetDiag().GetIndptr());
    solvers_.resize(agg_vertex_r.Rows());

    std::vector<int> marker(redist_.Rows(), -1);
    int num_aggs = agg_vertex.Rows();
    int num_vertices = agg_vertex.Cols();

    // Diagonal compensation
    std::vector<double> A_diag(A_r.Rows());
    {
        auto& diag_indptr = A_r.GetDiag().GetIndptr();
        auto& diag_indices = A_r.GetDiag().GetIndices();
        auto& diag_data = A_r.GetDiag().GetData();
        auto& offd_indptr = A_r.GetOffd().GetIndptr();
        auto& offd_indices = A_r.GetOffd().GetIndices();
        auto& offd_data = A_r.GetOffd().GetData();
        auto& col_map = A_r.GetColMap();
        auto& part = vertex_agg_r.GetDiag().GetIndices();

        const auto& A_ii = A_r.GetDiag().GetDiag();

        for (int row = 0; row < num_vertices; ++row)
        {
            for (int j = diag_indptr[row]; j < diag_indptr[row + 1]; ++j)
            {
                int col = diag_indices[j];

                if (col > row)
                {
                    int agg_i = part[row];
                    int agg_j = part[col];

                    if (agg_i != agg_j)
                    {
                        double tau = std::sqrt(A_ii[row] / A_ii[col]);
                        A_diag[row] += std::fabs(diag_data[j]) * tau;
                        A_diag[col] += std::fabs(diag_data[j]) / tau;

                        diag_data[j] = 0.0; // Not necessary
                    }
                }
            }

            for (int j = offd_indptr[row]; j < offd_indptr[row + 1]; ++j)
            {
                int agg_i = part[row];
                A_diag[row] += std::fabs(offd_data[j]);
                offd_data[j] = 0.0; // Not necessary
            }
        }
    }

    A_r.AddDiag(A_diag);

    if (A_r.GetMyId() == 0)
    {
        A_r.GetDiag().ToDense().Print("A diag:");
        A_r.GetOffd().ToDense().Print("A offd:");
    }

    for (int i = 0; i < num_aggs; ++i)
    {
        auto indices = agg_vertex_r.GetDiag().GetIndices(i);
        auto A_agg = A_r.GetDiag().GetSubMatrix(indices, indices, marker);

        DenseMatrix A_dense(A_agg.Rows(), A_agg.Cols());
        A_agg.ToDense(A_dense);

        A_dense.Print("A sub:");
        A_dense.Invert();
        A_dense.Print("A sub inv:");

        solvers_[i] = make_unique<DenseMatrix>(std::move(A_dense));
        block_op_.SetBlock(i, i, *solvers_[i]);
    }
}

ParMatrix ParBlockDiagComp::MakeRedistributor(const ParMatrix& agg_vertex)
{
    int diag_rows = agg_vertex.GetDiag().nnz();
    int diag_cols = agg_vertex.GetDiag().Cols();

    int offd_rows = agg_vertex.GetOffd().nnz();
    int offd_cols = agg_vertex.GetOffd().Cols();

    int total_rows = diag_rows + offd_rows;

    SparseMatrix<double> diag = SparseIdentity(total_rows, diag_cols);
    SparseMatrix<double> offd = SparseIdentity(total_rows, offd_cols, diag_rows);

    auto row_starts = GenerateOffsets(agg_vertex.GetComm(), total_rows);

    return ParMatrix(agg_vertex.GetComm(), row_starts, agg_vertex.GetColStarts(),
                     std::move(diag), std::move(offd), agg_vertex.GetColMap());
}

void ParBlockDiagComp::Mult(const linalgcpp::VectorView<double>& input,
                 linalgcpp::VectorView<double> output) const
{
    redist_.Mult(input, rhs_r_);
    block_op_.Mult(rhs_r_, sol_r_);
    redist_.MultAT(sol_r_, output);
}

} // namespace linalgcpp
