/*! @file */

#include "paroperator.hpp"

namespace linalgcpp
{

ParOperator::ParOperator()
    : comm_(0), myid_(-1), num_procs_(0), x_(nullptr), b_(nullptr)
{

}

ParOperator::ParOperator(MPI_Comm comm)
    : comm_(comm), x_(nullptr), b_(nullptr)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);
}

ParOperator::ParOperator(MPI_Comm comm, std::vector<HYPRE_Int> starts)
    : ParOperator(comm, starts, starts)
{

}

ParOperator::ParOperator(MPI_Comm comm, std::vector<HYPRE_Int> row_starts,
                         std::vector<HYPRE_Int> col_starts)
    : Operator(row_starts[1] - row_starts[0], col_starts[1] - col_starts[0]),
      comm_(comm), row_starts_(std::move(row_starts)), col_starts_(std::move(col_starts))
{
    assert(row_starts_.size() >= 3);
    assert(col_starts_.size() >= 3);

    HypreCreate();
}

ParOperator::ParOperator(const ParOperator& other)
    : linalgcpp::Operator(other), comm_(other.comm_),
      myid_(other.myid_), num_procs_(other.num_procs_),
      row_starts_(other.row_starts_), col_starts_(other.col_starts_),
      x_(nullptr), b_(nullptr)
{
    if (other.comm_ > 0 && myid_ >= 0)
    {
        HypreCreate();
    }
}

void ParOperator::HypreCreate()
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);

    InitVector(x_, row_starts_);
    InitVector(b_, col_starts_);
}

void ParOperator::InitVector(hypre_ParVector*& vect, std::vector<HYPRE_Int>& starts)
{
    assert(starts.size() >= 3);
    vect = hypre_ParVectorCreate(comm_, starts[2], starts.data());
    assert(vect);
    hypre_ParVectorSetPartitioningOwner(vect, 0);
    hypre_ParVectorSetDataOwner(vect, 1);
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(vect), 0);
    HYPRE_Complex tmp;
    hypre_VectorData(hypre_ParVectorLocalVector(vect)) = &tmp;
    hypre_ParVectorInitialize(vect);
}

void ParOperator::DestroyVector(hypre_ParVector*& vect)
{
    if (vect)
    {
        hypre_ParVectorDestroy(vect);
        vect = nullptr;
    }
}

void ParOperator::HypreDestroy()
{
    DestroyVector(x_);
    DestroyVector(b_);
}

ParOperator::~ParOperator()
{
    HypreDestroy();
}

void swap(ParOperator& lhs, ParOperator& rhs) noexcept
{
    swap(static_cast<linalgcpp::Operator&>(lhs),
         static_cast<linalgcpp::Operator&>(rhs));
    std::swap(lhs.comm_, rhs.comm_);
    std::swap(lhs.myid_, rhs.myid_);
    std::swap(lhs.num_procs_, rhs.num_procs_);
    std::swap(lhs.row_starts_, rhs.row_starts_);
    std::swap(lhs.col_starts_, rhs.col_starts_);
    std::swap(lhs.x_, rhs.x_);
    std::swap(lhs.b_, rhs.b_);
}

Vector<double> ParOperator::Mult(const linalgcpp::VectorView<double>& input) const
{
    Vector<double> output(Cols(), 0.0);

    Mult(input, output);

    return output;
}

int ParOperator::GlobalRows() const
{
    int size = 0.0;

    if (row_starts_.size() >= 3)
    {
        size = row_starts_[2];
    }

    return size;
}

int ParOperator::GlobalCols() const
{
    int size = 0.0;

    if (col_starts_.size() >= 3)
    {
        size = col_starts_[2];
    }

    return size;
}

const std::vector<HYPRE_Int>& ParOperator::GetRowStarts() const
{
    return row_starts_;
}

const std::vector<HYPRE_Int>& ParOperator::GetColStarts() const
{
    return col_starts_;
}

MPI_Comm ParOperator::GetComm() const
{
    return comm_;
}

int ParOperator::GetMyId() const
{
    return myid_;
}

int ParOperator::GetNumProcs() const
{
    return num_procs_;
}

double ParOperator::ParNorm(const VectorView<double>& x) const
{
    MPI_Comm comm = GetComm();

    Vector<double> Ax(x.size());
    this->Mult(x, Ax);

    double xAx = linalgcpp::ParMult(comm, x, Ax);

    return std::sqrt(xAx);
}

} // namespace linalgcpp

