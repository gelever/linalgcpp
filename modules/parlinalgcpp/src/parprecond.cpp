/*! @file */

#include "parprecond.hpp"

namespace linalgcpp
{

ParBlockDiagComp::ParBlockDiagComp(const ParMatrix& A_in, const ParMatrix& agg_vertex, int num_steps)
    : ParOperator(A_in), redist_(MakeRedistributer(agg_vertex)), num_steps_(num_steps),
      x_(redist_.Rows()), b_(redist_.Rows())
{
    //linalgcpp_verify(CheckSymmetric(A), "A not Symmetric!");

    ParMatrix redist_T = redist_.Transpose();
    ParMatrix agg_vertex_r = agg_vertex.Mult(redist_T);
    agg_vertex_r = 1.0;

    ParMatrix vertex_agg_r = agg_vertex_r.Transpose();
    A_r_ = redist_.Mult(A_in).Mult(redist_T);

    linalgcpp_parverify(agg_vertex_r.GetComm(), agg_vertex_r.GetOffd().nnz() == 0,
            "agg vertex not proper pattern!");

    block_op_ = BlockDiagOperator(agg_vertex_r.GetDiag().GetIndptr());
    solvers_.resize(agg_vertex_r.Rows());

    std::vector<int> marker(redist_.Rows(), -1);
    int num_aggs = agg_vertex_r.Rows();
    int num_vertices = A_r_.Rows();

    linalgcpp_verify(A_r_.Rows() == A_r_.Cols(), "A not square!");
    linalgcpp_verify(A_r_.Cols() == agg_vertex_r.Cols(), "A does not match agg_vertex!!");

    // Diagonal compensation
    Vector<double> A_diag(A_r_.Rows(), 0.0);
    const auto& diag_indptr = A_r_.GetDiag().GetIndptr();
    const auto& diag_indices = A_r_.GetDiag().GetIndices();
    const auto& diag_data = A_r_.GetDiag().GetData();

    const auto& offd_indptr = A_r_.GetOffd().GetIndptr();
    const auto& offd_indices = A_r_.GetOffd().GetIndices();
    const auto& offd_data = A_r_.GetOffd().GetData();

    const auto& col_map = A_r_.GetColMap();
    const auto& part = vertex_agg_r.GetDiag().GetIndices();

    const auto& A_ii = A_r_.GetDiag().GetDiag();

    auto comm_pkg = A_r_.MakeCommPkg();
    auto off_proc_diag = Broadcast(comm_pkg, A_r_.GetDiag().GetDiag());


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
                    double tau = std::sqrt(A_ii[row]) / std::sqrt(A_ii[col]);
                    A_diag[row] += std::fabs(diag_data[j]) * tau;
                    A_diag[col] += std::fabs(diag_data[j]) / tau;
                }
            }
        }

        for (int j = offd_indptr[row]; j < offd_indptr[row + 1]; ++j)
        {
            double tau = std::sqrt(A_ii[row]) / std::sqrt(off_proc_diag[offd_indices[j]]);
            A_diag[row] += std::fabs(offd_data[j]) * tau;
        }
    }

    Vector comp;

    for (int i = 0; i < num_aggs; ++i)
    {
        auto indices = agg_vertex_r.GetDiag().GetIndices(i);
        auto A_agg = A_r_.GetDiag().GetSubMatrix(indices, indices, marker);
        A_diag.GetSubVector(indices, comp);

        A_agg.AddDiag(comp.data());

        DenseMatrix A_dense(A_agg.Rows(), A_agg.Cols());
        A_agg.ToDense(A_dense);
        A_dense.Invert();

        solvers_[i] = make_unique<DenseMatrix>(std::move(A_dense));
        block_op_.SetBlock(i, i, solvers_[i]);
    }
}

ParMatrix ParBlockDiagComp::MakeRedistributor(const ParMatrix& agg_vertex)
{
    linalgcpp::linalgcpp_parverify(agg_vertex.nnz() == agg_vertex.GlobalCols(),
            "Agg_Vertex is not a proper pattern!");

    MPI_Comm comm = agg_vertex.GetComm();

    int diag_cols = agg_vertex.GetDiag().Cols();
    int offd_cols = agg_vertex.GetOffd().Cols();

    int diag_rows = agg_vertex.GetDiag().nnz();
    int offd_rows = agg_vertex.GetOffd().nnz();
    int total_rows = diag_rows + offd_rows;

    auto& vertices = agg_vertex.GetDiag().GetIndices();
    int num_vertices = vertices.size();

    SparseMatrix diag = SparseIdentity(total_rows, diag_cols, 0, diag_cols - num_vertices);
    SparseMatrix offd = SparseIdentity(total_rows, offd_cols, diag_rows);
    std::vector<int> col_map = agg_vertex.GetColMap();

    auto& indices = diag.GetIndices();
    std::copy(std::begin(vertices), std::end(vertices), std::begin(indices));

    auto starts = linalgcpp::GenerateOffsets(comm, total_rows);

    return ParMatrix(comm, std::move(starts), agg_vertex.GetColStarts(),
                     std::move(diag), std::move(offd), std::move(col_map));
}

void ParBlockDiagComp::Mult(const linalgcpp::VectorView<double>& input,
                 linalgcpp::VectorView<double> output) const
{
    redist_.Mult(input, b_);
    redist_.Mult(output, x_);

    Vector Ax(A_r_.Rows(), 0.0);
    Vector MAx(A_r_.Rows(), 0.0);

    for (int i = 0; i < num_steps_; ++i)
    {
        A_r_.Mult(x_, Ax);
        auto bAx = b_;
        bAx -= Ax;

        block_op_.Mult(bAx, MAx);
        x_ += MAx;
    }

    redist_.MultAT(x_, output);
}

} // namespace linalgcpp
