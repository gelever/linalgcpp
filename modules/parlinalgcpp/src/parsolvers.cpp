/*! @file */

#include "parsolvers.hpp"

namespace linalgcpp
{

ParSolver::ParSolver()
    : max_iter_(0), tol_(0), verbose_(false),
      create_(&no_create), destroy_(&no_destroy),
      setup_(&no_setup), solve_(&no_solve),
      solver_(nullptr)
{
}

ParSolver::ParSolver(ParMatrix A,
                     size_t max_iter, double tol, bool verbose,
                     HYPRE_Int (*create)(HYPRE_Solver*),
                     HYPRE_Int (*destroy)(HYPRE_Solver),
                     HYPRE_Int (*setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector),
                     HYPRE_Int (*solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector))
    : ParOperator(A.GetComm(), A.GetRowStarts(), A.GetColStarts()),
      max_iter_(max_iter), tol_(tol), verbose_(verbose),
      A_(std::move(A)), create_(create), destroy_(destroy), setup_(setup), solve_(solve),
      solver_(nullptr)
{
    assert(A_.Rows() == A_.Cols());
    assert(A_.GlobalRows() == A_.GlobalCols());
    assert(create);
    assert(destroy);
    assert(setup);
    assert(solve);

    HypreCreate();
}

ParSolver::~ParSolver()
{
    (*destroy_)(solver_);
}

ParSolver::ParSolver(const ParSolver& other)
    : ParOperator(other), max_iter_(other.max_iter_),
      tol_(other.tol_), verbose_(other.verbose_),
      A_(other.A_),
      create_(other.create_), destroy_(other.destroy_),
      setup_(other.setup_), solve_(other.solve_),
      solver_(nullptr)
{
    HypreCreate();
}

ParSolver::ParSolver(ParSolver&& other)
{
    swap(*this, other);
}

ParSolver& ParSolver::operator=(ParSolver other)
{
    swap(*this, other);

    return *this;
}

void swap(ParSolver& lhs, ParSolver& rhs) noexcept
{
    using std::swap;

    swap(static_cast<ParOperator&>(lhs), static_cast<ParOperator&>(rhs));

    swap(lhs.max_iter_, rhs.max_iter_);
    swap(lhs.tol_, rhs.tol_);
    swap(lhs.verbose_, rhs.verbose_);

    swap(lhs.A_, rhs.A_);
    swap(lhs.create_, rhs.create_);
    swap(lhs.destroy_, rhs.destroy_);
    swap(lhs.setup_, rhs.setup_);
    swap(lhs.solve_, rhs.solve_);
    swap(lhs.solver_, rhs.solver_);
}

void ParSolver::HypreCreate()
{
    (*create_)(&solver_);
}

void ParSolver::HypreSetup()
{
    const hypre_ParCSRMatrix* hypre_A = A_;
    hypre_ParCSRMatrix* hypre_A_c = const_cast<hypre_ParCSRMatrix*>(hypre_A);

    (*setup_)(solver_, hypre_A_c, b_, x_);
}

void ParSolver::Mult(const linalgcpp::VectorView<double>& input,
                     linalgcpp::VectorView<double> output) const
{
    hypre_VectorData(hypre_ParVectorLocalVector(b_)) = const_cast<double*>(std::begin(input));
    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = std::begin(output);

    output = 0.0;

    const hypre_ParCSRMatrix* hypre_A = A_;
    hypre_ParCSRMatrix* hypre_A_c = const_cast<hypre_ParCSRMatrix*>(hypre_A);

    (*solve_)(solver_, hypre_A_c, b_, x_);
}


BoomerAMG::BoomerAMG(ParMatrix A, size_t max_iter,
                     double tol, bool verbose)
    : ParSolver(std::move(A), max_iter, tol, verbose,
                &HYPRE_BoomerAMGCreate, &HYPRE_BoomerAMGDestroy,
                &HYPRE_BoomerAMGSetup, &HYPRE_BoomerAMGSolve)
{
    // Parameters mostly copied directly from mfem
    int coarsen_type = 10;   // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
    int agg_levels   = 1;    // number of aggressive coarsening levels
    double theta     = 0.25; // strength threshold: 0.25, 0.5, 0.8
    int interp_type  = 6;    // 6 = extended+i, 0 = classical
    int Pmax         = 4;    // max number of elements per row in P
    int relax_type   = 8;    // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
    int relax_sweeps = 1;    // relaxation sweeps on each level
    int max_levels   = 25;   // max number of levels in AMG hierarchy
    int print_level  = verbose_ ? 2 : 0;    // print AMG iterations? 1 = no, 2 = yes

    auto& solver_ = ParSolver::GetSolver();

    HYPRE_BoomerAMGSetCoarsenType(solver_, coarsen_type);
    HYPRE_BoomerAMGSetAggNumLevels(solver_, agg_levels);
    HYPRE_BoomerAMGSetRelaxType(solver_, relax_type);
    HYPRE_BoomerAMGSetNumSweeps(solver_, relax_sweeps);
    HYPRE_BoomerAMGSetStrongThreshold(solver_, theta);
    HYPRE_BoomerAMGSetInterpType(solver_, interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(solver_, Pmax);
    HYPRE_BoomerAMGSetMaxLevels(solver_, max_levels);
    HYPRE_BoomerAMGSetPrintLevel(solver_, print_level);
    HYPRE_BoomerAMGSetMaxIter(solver_, max_iter_);
    HYPRE_BoomerAMGSetTol(solver_, tol_);

    HypreSetup();
}

BoomerAMG::BoomerAMG(const BoomerAMG& other) noexcept
    : ParSolver(other)
{
}

BoomerAMG::BoomerAMG(BoomerAMG&& other) noexcept
{
    swap(*this, other);
}

BoomerAMG& BoomerAMG::operator=(BoomerAMG other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(BoomerAMG& lhs, BoomerAMG& rhs) noexcept
{
    swap(static_cast<ParSolver&>(lhs), static_cast<ParSolver&>(rhs));
}


ParaSails::ParaSails(ParMatrix A, bool verbose,
                     int reuse, int sym,
                     double load_bal, double threshold,
                     double filter, int max_levels)
    : ParSolver(std::move(A), 0, 0, verbose,
                &no_create, &HYPRE_ParaSailsDestroy,
                &HYPRE_ParaSailsSetup, &HYPRE_ParaSailsSolve)
{
    auto& solver_ = ParSolver::GetSolver();

    HYPRE_ParaSailsCreate(A_.GetComm(), &solver_);
    HYPRE_ParaSailsSetParams(solver_, threshold, max_levels);

    SetFilter(filter);
    SetSym(sym);
    SetLoadBal(load_bal);
    SetReuse(reuse);
    SetLogging(verbose);

    HypreSetup();
}

ParaSails::ParaSails(const ParaSails& other) noexcept
    : ParSolver(other)
{
}

ParaSails::ParaSails(ParaSails&& other) noexcept
{
    swap(*this, other);
}

ParaSails& ParaSails::operator=(ParaSails other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(ParaSails& lhs, ParaSails& rhs) noexcept
{
    swap(static_cast<ParSolver&>(lhs), static_cast<ParSolver&>(rhs));
}

void ParaSails::SetFilter(double filter)
{
    HYPRE_ParaSailsSetFilter(GetSolver(), filter);
}

void ParaSails::SetSym(int sym)
{
    HYPRE_ParaSailsSetSym(GetSolver(), sym);
}

void ParaSails::SetLoadBal(int load_bal)
{
    HYPRE_ParaSailsSetLoadbal(GetSolver(), load_bal);
}

void ParaSails::SetReuse(int reuse)
{
    HYPRE_ParaSailsSetReuse(GetSolver(), reuse);
}

void ParaSails::SetLogging(int logging)
{
    HYPRE_ParaSailsSetLogging(GetSolver(), logging);
}

ParCG::ParCG(ParMatrix A, size_t max_iter,
             double tol, bool verbose)
    : ParSolver(std::move(A), max_iter, tol, verbose,
                &HYPRE_ParCSRPCGCreate, &HYPRE_ParCSRPCGDestroy,
                &HYPRE_ParCSRPCGSetup, &HYPRE_ParCSRPCGSolve)
{
    auto& solver_ = ParSolver::GetSolver();

    HYPRE_PCGSetTol(solver_, tol_);
    HYPRE_PCGSetMaxIter(solver_, max_iter_);
    HYPRE_ParCSRPCGSetPrintLevel(solver_, verbose_);

    HypreSetup();
}

ParCG::ParCG(ParMatrix A, const ParSolver& M, size_t max_iter,
             double tol, bool verbose)
    : ParCG(std::move(A), max_iter, tol, verbose)
{
    auto& solver_ = ParSolver::GetSolver();

    HYPRE_ParCSRPCGSetPrecond(solver_,
                              M.GetSetupFn(), M.GetSolveFn(), M.GetSolver());
}


ParCG::ParCG(const ParCG& other) noexcept
    : ParSolver(other)
{
    HypreSetup();
}

ParCG::ParCG(ParCG&& other) noexcept
{
    swap(*this, other);
}

ParCG& ParCG::operator=(ParCG other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(ParCG& lhs, ParCG& rhs) noexcept
{
    swap(static_cast<ParSolver&>(lhs), static_cast<ParSolver&>(rhs));
}

void ParCG::EnableDiagScaling()
{
    auto& solver_ = ParSolver::GetSolver();

    HYPRE_ParCSRPCGSetPrecond(solver_,
                              (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScale,
                              (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                              nullptr);
}

ParDiagScale::ParDiagScale(ParMatrix A, size_t max_iter,
                           double tol, bool verbose)
    : ParSolver(std::move(A), max_iter, tol, verbose,
                &no_create, &no_destroy,
                &HYPRE_ParCSRDiagScaleSetup, &HYPRE_ParCSRDiagScale)
{
    HypreSetup();
}

ParDiagScale::ParDiagScale(const ParDiagScale& other) noexcept
    : ParSolver(other)
{
}

ParDiagScale::ParDiagScale(ParDiagScale&& other) noexcept
{
    swap(*this, other);
}

ParDiagScale& ParDiagScale::operator=(ParDiagScale other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(ParDiagScale& lhs, ParDiagScale& rhs) noexcept
{
    swap(static_cast<ParSolver&>(lhs), static_cast<ParSolver&>(rhs));
}

HYPRE_Int MySolve(void* solver, void* A, void* b, void* x )
{
    linalgcpp::Operator* op = (linalgcpp::Operator*) A;

    linalgcpp::VectorView<double> b_vect(((hypre_ParVector*)b)->local_vector->data, op->Rows());
    linalgcpp::VectorView<double> x_vect(((hypre_ParVector*)x)->local_vector->data, op->Rows());

    ((linalgcpp::Operator*) solver)->Mult(b_vect, x_vect);

    return 0;
}

HYPRE_Int MySetup(void* solver, void* A, void* b, void* x )
{
    return 0;
}

HYPRE_Int MyMatvec(void* matvec_data, HYPRE_Complex alpha,
                   void* A, void* x, HYPRE_Complex beta, void* y )
{
    linalgcpp::Operator* op = (linalgcpp::Operator*) A;

    linalgcpp::VectorView<double> x_vect(((hypre_ParVector*)x)->local_vector->data, op->Rows());
    linalgcpp::VectorView<double> y_vect(((hypre_ParVector*)y)->local_vector->data, op->Rows());

    op->Mult(x_vect, y_vect);

    return 0;
}

void* MyMatvecCreate(void* A, void* x)
{
    return nullptr;
}

HYPRE_Int MyMatvecDestroy(void* ptr)
{
    return 0;
}

std::vector<double> LOBPCG(const ParOperator& pmat, std::vector<linalgcpp::Vector<double>>& x,
                           const linalgcpp::Operator* precond, bool verbose)
{
    MPI_Comm comm = pmat.GetComm();
    int myid = pmat.GetMyId();

    int global_size = pmat.GlobalRows();
    int num_evects = x.size();
    std::vector<HYPRE_Int> row_starts = pmat.GetRowStarts();

    hypre_ParVector* vect = hypre_ParVectorCreate(comm, global_size, row_starts.data());

    hypre_ParVectorSetPartitioningOwner(vect, 0);
    hypre_ParVectorSetDataOwner(vect, 1);
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(vect), 0);
    HYPRE_Complex tmp;
    hypre_VectorData(hypre_ParVectorLocalVector(vect)) = &tmp;
    hypre_ParVectorInitialize(vect);
    hypre_VectorData(hypre_ParVectorLocalVector(vect)) = nullptr;

    HYPRE_Solver solver;
    HYPRE_MatvecFunctions matvec_fn;
    mv_InterfaceInterpreter interp;

    HYPRE_ParCSRSetupInterpreter(&interp);
    HYPRE_ParCSRSetupMatvec(&matvec_fn);
    HYPRE_LOBPCGCreate(&interp, &matvec_fn, &solver);

    matvec_fn.Matvec = MyMatvec;
    matvec_fn.MatvecCreate = MyMatvecCreate;
    matvec_fn.MatvecDestroy = MyMatvecDestroy;

    HYPRE_LOBPCGSetup(solver, (HYPRE_Matrix)&pmat, nullptr, nullptr);

    mv_MultiVectorPtr mv = mv_MultiVectorCreateFromSampleVector(&interp, num_evects, vect);

    mv_TempMultiVector* tmp_vect = (mv_TempMultiVector*)mv_MultiVectorGetData(mv);
    HYPRE_ParVector* vecs = (HYPRE_ParVector*)tmp_vect->vector;

    for (int i = 0; i < num_evects; ++i)
    {
        hypre_Vector* vec_i = vecs[i]->local_vector;
        std::copy(x[i].begin(), x[i].end(), vec_i->data);
    }

    std::vector<double> evals(num_evects, 0);

    assert(mv);

    int print_level = verbose;

    if (myid == 0)
    {
        HYPRE_LOBPCGSetPrintLevel(solver, print_level);
    }

    if (precond)
    {
        HYPRE_LOBPCGSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn)(MySolve),
                               (HYPRE_PtrToSolverFcn)(MySetup),
                               (HYPRE_Solver)precond);
    }

    HYPRE_LOBPCGSolve(solver, nullptr, mv, evals.data());

    for (int i = 0; i < num_evects; ++i)
    {
        hypre_Vector* vec_i = vecs[i]->local_vector;
        std::copy(vec_i->data, vec_i->data + x[i].size(), x[i].begin());
    }

    hypre_ParVectorDestroy(vect);
    mv_MultiVectorDestroy(mv);
    HYPRE_LOBPCGDestroy(solver);

    return evals;
}


std::vector<double> LOBPCG(const ParOperator& pmat, int num_evects, const linalgcpp::Operator* precond, bool verbose)
{
    std::vector<linalgcpp::Vector<double>> x(num_evects, linalgcpp::Vector<double>(pmat.Rows()));

    for (auto& x_i : x)
    {
        Randomize(x_i);
    }

    return LOBPCG(pmat, x, precond, verbose);
}

} // namespace linalgcpp
