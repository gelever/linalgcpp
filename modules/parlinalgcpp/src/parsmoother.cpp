/*! @file */

#include "parsmoother.hpp"

namespace linalgcpp
{
ParSmoother::ParSmoother()
    : buffer_shell_(nullptr)
{
}

ParSmoother::ParSmoother(ParMatrix A, SmoothType type,
                         int relax_times, double relax_weight,
                         double omega)
    : ParSolver(std::move(A)), type_(type), relax_times_(relax_times),
      relax_weight_(relax_weight), omega_(omega),
      buffer_(A_.Rows())
{
    InitVector(buffer_shell_, row_starts_);

    hypre_ParCSRComputeL1Norms(A_, static_cast<int>(type_), l1_norms_);
}

ParSmoother::ParSmoother(const ParSmoother& other)
    : ParSolver(other), type_(other.type_), relax_times_(other.relax_times_),
      relax_weight_(other.relax_weight_), omega_(other.omega_),
      buffer_(other.buffer_), buffer_shell_(nullptr), l1_norms_(other.l1_norms_)
{
    InitVector(buffer_shell_, row_starts_);
}

ParSmoother::ParSmoother(ParSmoother&& other)
{
    swap(*this, other);
}

ParSmoother& ParSmoother::operator=(ParSmoother other)
{
    swap(*this, other);

    return *this;
}

void swap(ParSmoother& lhs, ParSmoother& rhs) noexcept
{
    swap(static_cast<ParSolver&>(lhs), static_cast<ParSolver&>(rhs));

    std::swap(lhs.type_, rhs.type_);
    std::swap(lhs.relax_times_, rhs.relax_times_);
    std::swap(lhs.relax_weight_, rhs.relax_weight_);
    std::swap(lhs.omega_, rhs.omega_);
    swap(lhs.buffer_, rhs.buffer_);
    std::swap(lhs.buffer_shell_, rhs.buffer_shell_);
    std::swap(lhs.l1_norms_, rhs.l1_norms_);
}

ParSmoother::~ParSmoother()
{
    DestroyVector(buffer_shell_);
}

void ParSmoother::Mult(const linalgcpp::VectorView<double>& input,
                       linalgcpp::VectorView<double> output) const
{
    hypre_VectorData(hypre_ParVectorLocalVector(b_)) = const_cast<double*>(std::begin(input));
    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = std::begin(output);
    hypre_VectorData(hypre_ParVectorLocalVector(buffer_shell_)) = std::begin(buffer_);

    const hypre_ParCSRMatrix* hypre_A = GetMatrix();
    hypre_ParCSRMatrix* hypre_A_c = const_cast<hypre_ParCSRMatrix*>(hypre_A);

    int type = static_cast<int>(type_);
    double* l1_norms = l1_norms_.data();

    // Non user set parameters
    double max_eig_est = 1.0;
    double min_eig_est = 0.0;
    int poly_order = 1;
    double poly_fraction = 0.0;

    assert(l1_norms);
    assert(b_);
    assert(x_);
    assert(hypre_A_c);

    hypre_ParCSRRelax(hypre_A_c, b_, type, relax_times_, l1_norms, relax_weight_, omega_,
                      max_eig_est, min_eig_est, poly_order, poly_fraction,
                      x_, buffer_shell_, nullptr);
}

} // namespace linalgcpp
