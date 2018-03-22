#include "eigensolver.hpp"

extern "C"
{
    void dgesvd_(const char* jobu, const char* jobvt, const int* m,
                 const int* n, double* a, const int* lda, double* s,
                 double* u, const int* ldu, double* vt, const int* ldvt,
                 double* work, const int* lwork, int* info);

    void dsytrd_(char* uplo, const int* n, double* a, const int* lda,
                 double* d, double* e, double* tau, double* work,
                 int* lwork, int* info );

    void dsterf_(const int* n, double* d, double* e, int* info);

    void dstein_(const int* n, const double* d, const double* e,
                 int* m, const double* w, const int* iblock,
                 const int* isplit, double* z, const int* ldz,
                 double* work, int* iwork, int* ifailv,
                 int* info);

    void dormtr_(char* side, char* uplo, char* trans, const int* m,
                 const int* n, const double* a, const int* lda,
                 const double* tau, double* c, const int* ldc, double* work,
                 int* lwork, int* info);
}


namespace linalgcpp
{

EigenSolver::EigenSolver()
    : uplo_('U'), side_('L'), trans_('N'),
      abstol_(2 * std::numeric_limits<double>::min()),
      info_(0), alloc_size_(-1), lwork_(-1)
{
}

EigenPair EigenSolver::Solve(const SparseMatrix<double>& mat, double rel_tol, int max_evect)
{
    return Solve(mat.ToDense(), rel_tol, max_evect);
}

EigenPair EigenSolver::Solve(const DenseMatrix& mat, double rel_tol, int max_evect)
{
    EigenPair eigen_pair;
    Solve(mat, rel_tol, max_evect, eigen_pair);

    return eigen_pair;
}

void EigenSolver::Solve(const SparseMatrix<double>& mat, double rel_tol, int max_evect,
                        EigenPair& eigen_pair)
{
    Solve(mat.ToDense(), rel_tol, max_evect, eigen_pair);
}

void EigenSolver::Solve(const DenseMatrix& mat, double rel_tol, int max_evect,
                        EigenPair& eigen_pair)
{
    auto& evals = eigen_pair.first;
    auto& evects = eigen_pair.second;

    int n = mat.Rows();

    if (n < 1)
    {
        evals.resize(n);
        evects.Resize(0);
        printf("Zero Size EigenProblem!\n");
        return;
    }

    if (alloc_size_ < n)
    {
        AllocateWorkspace(n);
    }

    evals.resize(n);

    mat.CopyData(A_);
    double* a = A_.data();

    // Triangularize A = Q * T * Q^T
    dsytrd_(&uplo_, &n, a, &n, d_.data(), e_.data(), tau_.data(), work_.data(),
            &lwork_, &info_ );

    // d_ and e_ changed by dsterf_
    // copy since they are needed for dstein_
    std::copy_n(std::begin(d_), n, std::begin(evals));
    auto e_copy = e_;

    // Compute all eigenvalues
    dsterf_(&n, evals.data(), e_copy.data(), &info_);

    // Determine how many eigenvectors to be computed
    const double tol = evals.back() * rel_tol;

    if (max_evect < 0)
    {
        max_evect = n;
    }

    int m = 1;

    while (m < max_evect && evals[m] < tol)
    {
        ++m;
    }

    ifail_.resize(m);
    isplit_[0] = n;

    evects.Resize(n, m);
    //std::vector<double> evect_data(n * m);

    // Calculate Eigenvectors of T
    dstein_(&n, d_.data(), e_.data(),
            &m, evals.data(), iblock_.data(),
            isplit_.data(), evects.GetData(), &n,
            work_.data(), iwork_.data(), ifail_.data(),
            &info_);

    // Compute Q * (eigenvectors of T)
    dormtr_(&side_, &uplo_, &trans_, &n,
            &m, a, &n,
            tau_.data(), evects.GetData(), &n, work_.data(),
            &lwork_, &info_);

    // Should we keep all evals?
    evals.resize(n);

    //return std::make_tuple(std::move(evals), DenseMatrix(n, m, std::move(evect_data)));
}

void EigenSolver::AllocateWorkspace(int size)
{
    A_.resize(size * size, 0.0);
    alloc_size_ = size;

    int lwork = -1;
    double wkopt;

    // find max workspace between dsytrd_, dstein_ and dormtr_
    dsytrd_(&uplo_, &size, nullptr, &size,
            d_.data(), e_.data(), tau_.data(), &wkopt,
            &lwork, &info_ );

    // 5n is for dstein_
    lwork_ = std::max(5 * size, (int)wkopt);

    dormtr_(&side_, &uplo_, &trans_, &size,
            &size, nullptr, &size,
            tau_.data(), nullptr, &size, &wkopt,
            &lwork, &info_ );

    lwork_ = std::max(lwork_, (int)wkopt);

    work_.resize(lwork_);
    iwork_.resize(size);

    d_.resize(size);
    e_.resize(size);
    tau_.resize(size);

    iblock_.resize(size, 1);
    isplit_.resize(size, 0);
    iwork_.resize(size);
}

} // namespace linalgcpp
