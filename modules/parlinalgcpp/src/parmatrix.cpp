/*! @file */

#include "parmatrix.hpp"

namespace linalgcpp
{

ParMatrix::ParMatrix()
    : ParOperator(), A_(nullptr)
{

}

ParMatrix::ParMatrix(MPI_Comm comm,
                     linalgcpp::SparseMatrix<double> diag)
    : ParMatrix(comm, std::move(diag), linalgcpp::SparseMatrix<double>(), {})
{

}

ParMatrix::ParMatrix(MPI_Comm comm,
                     linalgcpp::SparseMatrix<double> diag,
                     linalgcpp::SparseMatrix<double> offd,
                     std::vector<HYPRE_Int> col_map)
    : ParOperator(comm), A_(nullptr), diag_(std::move(diag)),
      offd_(std::move(offd)), col_map_(std::move(col_map))
{
    auto starts = GenerateOffsets(comm, {diag_.Rows(), diag_.Cols()});
    assert(starts.size() == 2);

    std::swap(row_starts_, starts[0]);
    std::swap(col_starts_, starts[1]);

    InitVector(x_, row_starts_);
    InitVector(b_, col_starts_);

    Init();
}

ParMatrix::ParMatrix(MPI_Comm comm,
                     std::vector<HYPRE_Int> starts,
                     linalgcpp::SparseMatrix<double> diag)
    : ParMatrix(comm, starts, starts, std::move(diag))
{

}

ParMatrix::ParMatrix(MPI_Comm comm,
                     std::vector<HYPRE_Int> row_starts,
                     std::vector<HYPRE_Int> col_starts,
                     linalgcpp::SparseMatrix<double> diag)
    : ParMatrix(comm, std::move(row_starts), std::move(col_starts), std::move(diag),
                linalgcpp::SparseMatrix<double>(), {})
{

}

ParMatrix::ParMatrix(MPI_Comm comm,
                     std::vector<HYPRE_Int> row_starts,
                     std::vector<HYPRE_Int> col_starts,
                     linalgcpp::SparseMatrix<double> diag,
                     linalgcpp::SparseMatrix<double> offd,
                     std::vector<HYPRE_Int> col_map)
    : ParOperator(comm, std::move(row_starts), std::move(col_starts)),
      A_(nullptr), diag_(std::move(diag)), offd_(std::move(offd)), col_map_(std::move(col_map))
{
    Init();
}

void ParMatrix::Init()
{
    assert(static_cast<int>(row_starts_.size()) == 3);
    assert(static_cast<int>(col_starts_.size()) == 3);

    assert(row_starts_[0] >= 0);
    assert(row_starts_[1] <= row_starts_[2]);
    assert(col_starts_[0] >= 0);
    assert(col_starts_[1] <= col_starts_[2]);

    int row_size = row_starts_[1] - row_starts_[0];
    int col_size = col_starts_[1] - col_starts_[0];
    int offd_size = col_map_.size();

    rows_ = row_size;
    cols_ = col_size;

    if (offd_.Rows() == 0 && offd_.Cols() == 0)
    {
        offd_ = linalgcpp::SparseMatrix<double>(rows_, 0);
    }

    assert(diag_.Rows() == row_size);
    assert(diag_.Cols() == col_size);
    assert(offd_.Rows() == row_size);
    assert(offd_.Cols() == offd_size);

    for (int i = 0; i < offd_size; ++i)
    {
        assert(col_map_[i] >= 0 && col_map_[i] < GlobalCols());
        assert(col_map_[i] >= col_starts_[1] || col_map_[i] < col_starts_[0]);

        if (i > 0)
        {
            assert(col_map_[i] > col_map_[i - 1]);
        }

        if (i < offd_size - 1)
        {
            assert(col_map_[i] < col_map_[i + 1]);
        }
    }

    HypreCreate();
}


ParMatrix::~ParMatrix() noexcept
{
    HypreDestroy();
}

ParMatrix::ParMatrix(const ParMatrix& other) noexcept
    : ParOperator(other),
      A_(nullptr), diag_(other.diag_), offd_(other.offd_),
      col_map_(other.col_map_)
{
    if (other.A_)
    {
        HypreCreate();
    }
}

void swap(ParMatrix& lhs, ParMatrix& rhs) noexcept
{
    swap(static_cast<ParOperator&>(lhs),
         static_cast<ParOperator&>(rhs));

    std::swap(lhs.A_, rhs.A_);
    std::swap(lhs.col_map_, rhs.col_map_);

    swap(lhs.diag_, rhs.diag_);
    swap(lhs.offd_, rhs.offd_);
}

ParMatrix::ParMatrix(ParMatrix&& other) noexcept
    : A_(nullptr)
{
    swap(*this, other);
}
ParMatrix& ParMatrix::operator=(ParMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}

void ParMatrix::HypreCreate()
{
    HypreDestroy();

    A_ = hypre_ParCSRMatrixCreate(comm_, GlobalRows(), GlobalCols(),
                                  row_starts_.data(), col_starts_.data(),
                                  offd_.Cols(), diag_.nnz(), offd_.nnz());

    hypre_ParCSRMatrixOwnsData(A_) = false;
    hypre_ParCSRMatrixOwnsRowStarts(A_) = false;
    hypre_ParCSRMatrixOwnsColStarts(A_) = false;

    hypre_CSRMatrixOwnsData(A_->diag) = false;
    hypre_CSRMatrixData(A_->diag) = diag_.GetData().data();
    hypre_CSRMatrixI(A_->diag) = diag_.GetIndptr().data();
    hypre_CSRMatrixJ(A_->diag) = diag_.GetIndices().data();

    // TODO(gelever): This leaks, figure out what to delete
    //hypre_CSRMatrixSetRownnz(A_->diag);

    hypre_CSRMatrixOwnsData(A_->offd) = false;
    hypre_CSRMatrixData(A_->offd) = offd_.GetData().data();
    hypre_CSRMatrixI(A_->offd) = offd_.GetIndptr().data();
    hypre_CSRMatrixJ(A_->offd) = offd_.GetIndices().data();
    hypre_CSRMatrixSetRownnz(A_->offd);

    hypre_ParCSRMatrixColMapOffd(A_) = col_map_.data();

    hypre_ParCSRMatrixSetNumNonzeros(A_);
    hypre_CSRMatrixReorder(A_->diag);
    hypre_MatvecCommPkgCreate(A_);
}

void ParMatrix::HypreDestroy()
{
    if (A_)
    {
        hypre_MatvecCommPkgDestroy(A_->comm_pkg);
        hypre_TFree(A_->offd->rownnz);
        hypre_TFree(A_->diag);
        hypre_TFree(A_->offd);
        hypre_ParCSRMatrixDestroy(A_);

        A_ = nullptr;
    }
}

linalgcpp::Vector<double> ParMatrix::Mult(const linalgcpp::VectorView<double>& input) const
{
    linalgcpp::Vector<double> output(diag_.Rows());
    Mult(input, output);

    return output;
}

void ParMatrix::Mult(const linalgcpp::VectorView<double>& input,
                     linalgcpp::VectorView<double> output) const
{
    assert(input.size() == diag_.Cols());
    assert(output.size() == diag_.Rows());

    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = const_cast<double*>(std::begin(input));
    hypre_VectorData(hypre_ParVectorLocalVector(b_)) = std::begin(output);

    output = 0.0;

    const double alpha = 1.0;
    const double beta = 0.0;

    hypre_ParCSRMatrixMatvec(alpha, A_, x_, beta, b_);
}

linalgcpp::Vector<double> ParMatrix::MultAT(const linalgcpp::VectorView<double>& input) const
{
    linalgcpp::Vector<double> output(diag_.Cols());
    MultAT(input, output);

    return output;
}

void ParMatrix::MultAT(const linalgcpp::VectorView<double>& input,
                       linalgcpp::VectorView<double> output) const
{
    assert(input.size() == diag_.Rows());
    assert(output.size() == diag_.Cols());

    hypre_VectorData(hypre_ParVectorLocalVector(b_)) = const_cast<double*>(std::begin(input));
    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = std::begin(output);

    output = 0.0;

    const double alpha = 1.0;
    const double beta = 0.0;

    hypre_ParCSRMatrixMatvecT(alpha, A_, b_, beta, x_);
}

ParVector ParMatrix::Mult(const ParVector& input) const
{
    ParVector output(comm_, GlobalRows(), row_starts_);
    Mult(input, output);

    return output;
}

void ParMatrix::Mult(const ParVector& input, ParVector& output) const
{
    assert(input.size() == diag_.Cols());
    assert(output.size() == diag_.Rows());

    assert(input.GlobalSize() == GlobalCols());
    assert(output.GlobalSize() == GlobalRows());

    output = 0.0;

    const double alpha = 1.0;
    const double beta = 0.0;

    hypre_ParCSRMatrixMatvec(alpha, A_, input.pvect_, beta, output.pvect_);
}

ParVector ParMatrix::MultAT(const ParVector& input) const
{
    ParVector output(comm_, GlobalCols(), col_starts_);
    MultAT(input, output);

    return output;
}

void ParMatrix::MultAT(const ParVector& input, ParVector& output) const
{
    assert(input.size() == diag_.Rows());
    assert(output.size() == diag_.Cols());

    assert(input.GlobalSize() == GlobalRows());
    assert(output.GlobalSize() == GlobalCols());

    output = 0.0;

    const double alpha = 1.0;
    const double beta = 0.0;

    hypre_ParCSRMatrixMatvecT(alpha, A_, input.pvect_, beta, output.pvect_);
}

ParMatrix& ParMatrix::operator*=(double val)
{
    diag_ *= val;
    offd_ *= val;

    return *this;
}

ParMatrix& ParMatrix::operator=(double val)
{
    diag_ = val;
    offd_ = val;

    return *this;
}

ParMatrix ParMatrix::operator-() const
{
    ParMatrix copy(*this);
    copy *= -1.0;

    return copy;
}

void ParMatrix::Print(const std::string& label, std::ostream& out) const
{
    int num_procs = GetNumProcs();

    for (int i = 0; i < num_procs; ++i)
    {
        if (GetMyId() == i)
        {
            out << label << "\n";

            diag_.Print("Diag:", out);
            offd_.Print("Offd:", out);

            using linalgcpp::operator<<;
            out << "ColMap:" << col_map_;
            out.flush();
        }
        MPI_Barrier(GetComm());
    }
}

void ParMatrix::AddDiag(double diag_val)
{
    diag_.AddDiag(diag_val);
}

void ParMatrix::AddDiag(const std::vector<double>& diag_vals)
{
    diag_.AddDiag(diag_vals);
}

ParCommPkg ParMatrix::MakeCommPkg() const
{
    linalgcpp_assert(A_ != NULL, "ParMatrix::A_ not valid!");

    return ParCommPkg(A_);
}

double ParMatrix::MaxNorm() const
{
    double local_max = std::max(diag_.MaxNorm(), offd_.MaxNorm());
    double global_max = 0.0;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  comm_);

    return global_max;
}

int ParMatrix::nnz() const
{
    return A_ ? A_->num_nonzeros : 0;
}

void ParMatrix::ScaleRows(const linalgcpp::SparseMatrix<double>& values)
{
    diag_.ScaleRows(values);
    offd_.ScaleRows(values);
}

void ParMatrix::InverseScaleRows(const linalgcpp::SparseMatrix<double>& values)
{
    diag_.InverseScaleRows(values);
    offd_.InverseScaleRows(values);
}

void ParMatrix::ScaleRows(const std::vector<double>& values)
{
    diag_.ScaleRows(values);
    offd_.ScaleRows(values);
}

void ParMatrix::InverseScaleRows(const std::vector<double>& values)
{
    diag_.InverseScaleRows(values);
    offd_.InverseScaleRows(values);
}

void ParMatrix::EliminateRow(int index)
{
    diag_.EliminateRow(index);
    offd_.EliminateRow(index);
}

int ParMatrix::RowSize(int row) const
{
    return diag_.RowSize(row) + offd_.RowSize(row);
}

void ParMatrix::EliminateZeros(double tol, bool keep_diag)
{
    diag_.EliminateZeros(tol, keep_diag);
    offd_.EliminateZeros(tol);

    HypreCreate();
}

} // namespace linalgcpp
