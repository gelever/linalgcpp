#include "graphtools.hpp"
#include "test_input_graphs.hpp"

using namespace linalgcpp;

ParMatrix AggReDistributer(const ParMatrix& agg_vertex);
Vector<double> SmoothVector(const ParMatrix& A, int max_iter);
void CheckPartition(const ParMatrix& agg_vertex, const std::string& label = "Partition");
void VerifyRowSizeOne(const ParMatrix& A);

int main(int argc, char** argv)
{
    MpiSession mpi(argc, argv);

    auto vertex_vertex = ReadMTX("sts4098.mtx");
    linalgcpp_verify(CheckSymmetric(vertex_vertex), "Input not Symmetric!");

    auto proc_part = Partition(vertex_vertex, mpi.num_procs, 1.0, false);
    ParMatrix A = linalgcpp::ParSplit(mpi.comm, vertex_vertex, proc_part);

    auto smooth_vector = SmoothVector(A, 10);

    ParMatrix W(mpi.comm, SparseMatrix<double>(smooth_vector.data()));

    A = W.Mult(A).Mult(W);

    ParMatrix PT = ParPartition(A, 2);
    ParMatrix P = PT.Transpose();

    ParMatrix redist = AggReDistributer(PT);
    ParMatrix redist_T = redist.Transpose();
    ParMatrix PT_r = PT.Mult(redist_T);

    CheckPartition(PT, "PT");
    CheckPartition(PT_r, "PT_r");
    CheckPartition(redist_T, "redist_T");
    CheckPartition(redist, "redist");
}

ParMatrix AggReDistributer(const ParMatrix& agg_vertex)
{
    linalgcpp::linalgcpp_verify(agg_vertex.nnz() == agg_vertex.GlobalCols(),
            "Agg_Vertex is not a proper pattern!");

    MPI_Comm comm = agg_vertex.GetComm();

    int diag_cols = agg_vertex.GetDiag().Cols();
    int offd_cols = agg_vertex.GetOffd().Cols();

    int diag_rows = agg_vertex.GetDiag().nnz();
    int offd_rows = agg_vertex.GetOffd().nnz();
    int total_rows = diag_rows + offd_rows;

    auto& vertices = agg_vertex.GetDiag().GetIndices();
    int num_vertices = vertices.size();

    auto diag = SparseIdentity(total_rows, diag_cols, 0, diag_cols - num_vertices);
    auto offd = SparseIdentity(total_rows, offd_cols, diag_rows);
    std::vector<int> col_map = agg_vertex.GetColMap();

    auto& indices = diag.GetIndices();
    std::copy(std::begin(vertices), std::end(vertices), std::begin(indices));

    auto starts = linalgcpp::GenerateOffsets(comm, total_rows);

    return ParMatrix(comm, std::move(starts), agg_vertex.GetColStarts(),
                     std::move(diag), std::move(offd), std::move(col_map));
}

Vector<double> SmoothVector(const ParMatrix& A, int max_iter)
{
    int num_rows = A.Rows();

    if (max_iter <= 0)
    {
        Vector<double> w(num_rows, 1.0);
        w /= A.ParNorm(w);

        return w;
    }

    Vector<double> w(num_rows);
    Vector<double> zero(num_rows, 0);

    int seed = A.GetMyId() + 1;
    Randomize(w, -1.0, 1.0, seed);

    ParSmoother smoother(A, linalgcpp::SmoothType::L1_GS, max_iter);
    smoother.Mult(zero, w);

    VerifyFinite(w);

    Vector<double> Aw(num_rows, 0.0);
    A.Mult(w, Aw);
    double w_norm = ParMult(A.GetComm(), w, Aw);

    if (std::fabs(w_norm) < 1e-12)
    {
        throw std::runtime_error("initial W is sufficiently smooth!" + std::to_string(w_norm));
    }

    w /= std::sqrt(w_norm);

    VerifyFinite(w);

    assert(std::fabs(1.0 - A.ParNorm(w)) < 1e-9);

    return w;
}

void VerifyRowSizeOne(const ParMatrix& A)
{
    int num_rows = A.Rows();
    int is_bad = 0;

    for (int i = 0; i < num_rows; ++i)
    {
        int diag_row_size = A.GetDiag().RowSize(i);
        int offd_row_size = A.GetOffd().RowSize(i);
        int row_size = diag_row_size + offd_row_size;

        is_bad += (row_size != 1);
    }

    linalgcpp_parverify(A.GetComm(), is_bad == 0,
            std::to_string(A.GetMyId()) + " Row Size Not One! : " + std::to_string(is_bad) );
}

void CheckPartition(const ParMatrix& agg_vertex, const std::string& label)
{
    ParMatrix vertex_agg = RemoveLargeEntries(agg_vertex.Transpose());

    ParPrint(agg_vertex.GetMyId(), printf("Checking %s\t(%d %d, %d)\t",
                label.c_str(), agg_vertex.GlobalRows(), agg_vertex.GlobalCols(), agg_vertex.nnz() ));
    VerifyRowSizeOne(vertex_agg);
    ParPrint(agg_vertex.GetMyId(), printf("Good Partition\n"));
}
