/*! @file */

#include "parvector.hpp"

namespace linalgcpp
{

ParVector::ParVector()
    : Vector(0), comm_(0), pvect_(nullptr)
{
}

ParVector::ParVector(MPI_Comm comm, const std::vector<HYPRE_Int>& part)
    : ParVector(comm, 0, part)
{
}

ParVector::ParVector(MPI_Comm comm, double val, const std::vector<HYPRE_Int>& part)
    : Vector(part[1] - part[0], val), comm_(comm), pvect_(nullptr), partition_(part)
{
    printf("Parvector: %d %d - > %d\n", part[1], part[0], size());
    Init();
}

ParVector::ParVector(MPI_Comm comm, const linalgcpp::VectorView<double>& vect,
                     const std::vector<HYPRE_Int>& part)
    : Vector(vect), comm_(comm), pvect_(nullptr), partition_(part)
{
    Init();
}

void ParVector::Init()
{
    assert(partition_.size() >= 2);

    assert(partition_[0] >= 0);
    assert(partition_[1] >= 0);

    global_size_ = partition_[2];

    const int local_size = partition_[1] - partition_[0];
    assert(linalgcpp::Vector<double>::size() == local_size);

    assert(partition_[0] < global_size_);
    assert(partition_[1] <= global_size_);

    assert(partition_[1] >= partition_[0]);

    HypreCreate();
}



ParVector::ParVector(const ParVector& other)
    : linalgcpp::Vector<double>(other), comm_(other.comm_), pvect_(nullptr), partition_(other.partition_),
      global_size_(other.global_size_)
{
    HypreCreate();
}

ParVector::ParVector(ParVector&& other)
    : comm_(0), pvect_(nullptr), partition_(), global_size_(0)
{
    swap(*this, other);
}

ParVector& ParVector::operator=(ParVector other)
{
    swap(*this, other);

    return *this;
}

void swap(ParVector& lhs, ParVector& rhs) noexcept
{
    linalgcpp::swap(static_cast<linalgcpp::Vector<double>&>(lhs),
                    static_cast<linalgcpp::Vector<double>&>(rhs));
    std::swap(lhs.comm_, rhs.comm_);
    std::swap(lhs.pvect_, rhs.pvect_);
    std::swap(lhs.partition_, rhs.partition_);
    std::swap(lhs.global_size_, rhs.global_size_);
}

int ParVector::GlobalSize() const
{
    return global_size_;
}

ParVector::~ParVector() noexcept
{
    HypreDestroy();
}

// Modified from MFEM's HypreParVector constructor
void ParVector::HypreCreate()
{
    HypreDestroy();

    pvect_ = hypre_ParVectorCreate(comm_, global_size_, partition_.data());
    assert(pvect_);
    hypre_ParVectorSetPartitioningOwner(pvect_, 0);
    hypre_ParVectorSetDataOwner(pvect_, 1);
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(pvect_), 0);
    HYPRE_Complex tmp;
    hypre_VectorData(hypre_ParVectorLocalVector(pvect_)) = &tmp;
    hypre_ParVectorInitialize(pvect_);
    hypre_VectorData(hypre_ParVectorLocalVector(pvect_)) = linalgcpp::Vector<double>::begin();
}

void ParVector::HypreDestroy()
{
    if (pvect_)
    {
        hypre_ParVectorDestroy(pvect_);
        pvect_ = nullptr;
    }
}

ParVector& operator+=(ParVector& lhs, const ParVector& rhs)
{
    lhs += static_cast<const linalgcpp::Vector<double>&>(rhs);

    return lhs;
}

ParVector& operator-=(ParVector& lhs, const ParVector& rhs)
{
    lhs -= static_cast<const linalgcpp::Vector<double>&>(rhs);

    return lhs;
}

ParVector& operator*=(ParVector& lhs, const ParVector& rhs)
{
    lhs *= static_cast<const linalgcpp::Vector<double>&>(rhs);

    return lhs;
}
ParVector& operator/=(ParVector& lhs, const ParVector& rhs)
{
    lhs /= static_cast<const linalgcpp::Vector<double>&>(rhs);

    return lhs;
}

ParVector& ParVector::operator=(double val)
{
    this->Vector<double>::operator=(val);

    return *this;
}

double ParVector::Mult(const VectorView<double>& vect) const
{
    double local_sum = linalgcpp::VectorView<double>::Mult(vect);

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm_);

    return global_sum;
}

double ParVector::LocalSum() const
{
    return linalgcpp::Sum(*this);
}

double ParVector::GlobalSum() const
{
    double local_sum = LocalSum();

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm_);

    int myid;
    MPI_Comm_rank(comm_, &myid);

    return global_sum;
}

void SubAvg(ParVector& vect)
{
    vect -= vect.GlobalSum() / vect.GlobalSize();
}


/*
    double ParVector::Mult(const ParVector& vect) const
    {
    double local_sum = linalgcpp::operator*<double, double>(*this, vect);

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm_);

    return global_sum;
    }

    double ParVector::InnerProduct(const ParVector& vect) const
    {
    return Mult(vect);
    }
*/

double ParVector::L2Norm() const
{
    return std::sqrt(Mult(*this));
}

double Mult(const ParVector& lhs, const ParVector& rhs)
{
    return lhs.Mult(rhs);
}

double ParL2Norm(MPI_Comm comm, const linalgcpp::VectorView<double>& vect)
{
    double local_sum = vect.Mult(vect);

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    return std::sqrt(global_sum);
}

double L2Norm(const ParVector& vect)
{
    return vect.L2Norm();
}

double ParMult(MPI_Comm comm, const linalgcpp::VectorView<double>& lhs, const linalgcpp::VectorView<double>& rhs)
{
    double local_sum = lhs.Mult(rhs);

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_sum;
}

} // namespace linalgcpp
