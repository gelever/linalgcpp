/*! @file */

#include "sparsesolve.hpp"

namespace linalgcpp
{

SparseSolver::SparseSolver(linalgcpp::SparseMatrix<double> A)
    : Operator(A), A_(std::move(A)), numeric_(nullptr),
      control_(UMFPACK_CONTROL, 0), info_(UMFPACK_INFO, 0)
{
    assert(A_.Rows() == A_.Cols());

    // UMFPACK doesn't like empty matrices
    if (A_.nnz() > 0)
    {
        A_.SortIndices();

        Init();
    }
}

SparseSolver::SparseSolver(const SparseSolver& other) noexcept
    : Operator(other), A_(other.A_), control_(other.control_), info_(other.info_)
{
    if (A_.nnz() > 0)
    {
        Init();
    }
}

SparseSolver::SparseSolver(SparseSolver&& other) noexcept
{
    swap(*this, other);
}

SparseSolver& SparseSolver::operator=(SparseSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(SparseSolver& lhs, SparseSolver& rhs) noexcept
{
    swap(static_cast<linalgcpp::Operator&>(lhs), static_cast<linalgcpp::Operator&>(rhs));
    swap(lhs.A_, rhs.A_);

    std::swap(lhs.numeric_, rhs.numeric_);
    std::swap(lhs.control_, rhs.control_);
    std::swap(lhs.info_, rhs.info_);
}

void SparseSolver::Init()
{
    assert(A_.nnz());

    const int* indptr = A_.GetIndptr().data();
    const int* indices = A_.GetIndices().data();
    const double* data = A_.GetData().data();

    int rows = A_.Rows();
    int cols = A_.Cols();
    
    umfpack_di_defaults(control_.data());

    void* symbolic = nullptr;
    int symb_status = umfpack_di_symbolic(rows, cols, indptr, indices, data,
            &symbolic, control_.data(), info_.data());

    assert(symb_status == 0);

    int num_status = umfpack_di_numeric(indptr, indices, data,
            symbolic, &numeric_, control_.data(), info_.data());

    assert(num_status == 0);

    umfpack_di_free_symbolic(&symbolic);
}

SparseSolver::~SparseSolver()
{
    if (numeric_)
    {
        umfpack_di_free_numeric(&numeric_);
        numeric_ = nullptr;
    }
}

void SparseSolver::Mult(const linalgcpp::VectorView<double>& input,
        linalgcpp::VectorView<double> output) const
{
    assert(input.size() == A_.Cols());
    assert(output.size() == A_.Rows());

    if (!numeric_)
    {
        output = 0.0;
        return;
    }

    const int* indptr = A_.GetIndptr().data();
    const int* indices = A_.GetIndices().data();
    const double* data = A_.GetData().data();

    const double* in = input.begin();
    double* out = output.begin();

    int status = umfpack_di_solve(UMFPACK_At, indptr, indices, data,
                                  out, in, numeric_, control_.data(), info_.data());

    umfpack_di_report_info(control_.data(), info_.data());

    assert(status == 0);
}

void SparseSolver::MultAT(const linalgcpp::VectorView<double>& input,
        linalgcpp::VectorView<double> output) const
{
    assert(input.size() == A_.Rows());
    assert(output.size() == A_.Cols());

    if (!numeric_)
    {
        output = 0.0;
        return;
    }

    const int* indptr = A_.GetIndptr().data();
    const int* indices = A_.GetIndices().data();
    const double* data = A_.GetData().data();

    const double* in = input.begin();
    double* out = output.begin();

    int status = umfpack_di_solve(UMFPACK_A, indptr, indices, data,
                                  out, in, numeric_, control_.data(), info_.data());

    umfpack_di_report_info(control_.data(), info_.data());

    assert(status == 0);
}

} // namespace linalgcpp
