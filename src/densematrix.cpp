#include "densematrix.hpp"

extern "C"
{
    void dgemm_(const char* transA, const char* transB,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb, const double* beta,
                double* C, const int* ldc);

    void dsyevd_(const char* jobz, const char* uplo, const int* n,
                 double* A, const int* lda, double* w, double* work,
                 const int* lwork, int* iwork, const int* liwork, int* info);

    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
                 double* A, const int* lda, double* S, double* U, const int* ldu,
                 double* VT, const int* ldvt, double* work, const int* lwork, int* info);

    void dgeqp3_(const int* m, const int* n, double* A, const int* lda,
                 int* jpvt, double* tau, double* work, const int* lwork, int* info);

    void dorgqr_(const int* m, const int* n, const int* k, double* A,
                 const int* lda, double* tau, double* work, const int* lwork,
                 int* info);
}

namespace linalgcpp
{

DenseMatrix::DenseMatrix()
    : DenseMatrix(0)
{
}

DenseMatrix::DenseMatrix(size_t size)
    : DenseMatrix(size, size)
{
}

DenseMatrix::DenseMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0)
{
}

DenseMatrix::DenseMatrix(size_t rows, size_t cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data)
{
    assert(data.size() == rows * cols);
}

DenseMatrix::DenseMatrix(const DenseMatrix& other) noexcept
: rows_(other.rows_), cols_(other.cols_), data_(other.data_)
{
}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
{
    Swap(*this, other);
}

DenseMatrix& DenseMatrix::operator=(DenseMatrix other) noexcept
{
    Swap(*this, other);

    return *this;
}

void Swap(DenseMatrix& lhs, DenseMatrix& rhs)
{
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.cols_, rhs.cols_);
    std::swap(lhs.data_, rhs.data_);
}


void DenseMatrix::Print(const std::string& label, std::ostream& out, int width, int precision) const
{
    out << label << "\n";

    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            out << std::setw(width) << std::setprecision(precision)
                << std::fixed << (*this)(i, j);
                //<< std::defaultfloat << (*this)(i, j);
        }

        out << "\n";
    }

    out << "\n";
}

DenseMatrix DenseMatrix::Transpose() const
{
    DenseMatrix transpose(cols_, rows_);

    Transpose(transpose);

    return transpose;
}

void DenseMatrix::Transpose(DenseMatrix& transpose) const
{
    assert(transpose.Rows() == cols_);
    assert(transpose.Cols() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            transpose(j, i) = (*this)(i, j);
        }
    }
}

void DenseMatrix::Mult(const VectorView<double>& input, VectorView<double>& output) const
{
    Mult<double, double>(input, output);
}

void DenseMatrix::MultAT(const VectorView<double>& input, VectorView<double>& output) const
{
    MultAT<double, double>(input, output);
}

DenseMatrix DenseMatrix::Mult(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.Cols());
    Mult(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultAT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.Cols());
    MultAT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultBT(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.Rows());
    MultBT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultABT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.Rows());
    MultABT(input, output);

    return output;
}

void DenseMatrix::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(cols_ == input.Rows());
    assert(rows_ == output.Rows());
    assert(input.Cols() == output.Cols());

    bool AT = false;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultAT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(rows_ == input.Rows());
    assert(cols_ == output.Rows());
    assert(input.Cols() == output.Cols());

    bool AT = true;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultBT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(cols_ == input.Cols());
    assert(rows_ == output.Rows());
    assert(input.Rows() == output.Cols());

    bool AT = false;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultABT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(rows_ == input.Cols());
    assert(cols_ == output.Rows());
    assert(input.Rows() == output.Cols());

    bool AT = true;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const
{
    char transA = AT ? 'T' : 'N';
    char transB = BT ? 'T' : 'N';
    int m = AT ? cols_ : rows_;
    int n = BT ? input.Rows() : input.Cols();
    int k = AT ? rows_ : cols_;

    double alpha = 1.0;
    const double* A = data_.data();
    int lda = rows_;
    const double* B = input.data_.data();
    int ldb = input.Rows();
    double beta = 0.0;
    double* c = output.data_.data();
    int ldc = output.Rows();

    dgemm_(&transA, &transB, &m, &n, &k,
           &alpha, A, &lda, B, &ldb,
           &beta, c, &ldc);
}

DenseMatrix& DenseMatrix::operator-=(const DenseMatrix& other)
{
    assert(rows_ == other.Rows());
    assert(cols_ == other.Cols());

    const size_t nnz = rows_ * cols_;

    for (size_t i = 0; i < nnz; ++i)
    {
        data_[i] -= other.data_[i];
    }

    return *this;
}

DenseMatrix& DenseMatrix::operator+=(const DenseMatrix& other)
{
    assert(rows_ == other.Rows());
    assert(cols_ == other.Cols());

    const size_t nnz = rows_ * cols_;

    for (size_t i = 0; i < nnz; ++i)
    {
        data_[i] += other.data_[i];
    }

    return *this;
}

DenseMatrix operator+(DenseMatrix lhs, const DenseMatrix& rhs)
{
    return lhs += rhs;
}

DenseMatrix operator-(DenseMatrix lhs, const DenseMatrix& rhs)
{
    return lhs -= rhs;
}

DenseMatrix& DenseMatrix::operator*=(double val)
{
    for (auto& i : data_)
    {
        i *= val;
    }

    return *this;
}

DenseMatrix operator*(DenseMatrix lhs, double val)
{
    return lhs *= val;
}

DenseMatrix operator*(double val, DenseMatrix rhs)
{
    return rhs *= val;
}

DenseMatrix& DenseMatrix::operator/=(double val)
{
    assert(val != 0);

    for (auto& i : data_)
    {
        i /= val;
    }

    return *this;
}

DenseMatrix operator/(DenseMatrix lhs, double val)
{
    return lhs /= val;
}

DenseMatrix& DenseMatrix::operator=(double val)
{
    std::fill(begin(data_), end(data_), val);

    return *this;
}

bool DenseMatrix::operator==(const DenseMatrix& other) const
{
    if (other.Rows() != rows_ || other.Cols() != cols_)
    {
        return false;
    }

    constexpr double tol = 1e-12;

    for (size_t j = 0; j < cols_; ++j)
    {
        for (size_t i = 0; i < rows_; ++i)
        {
            if (std::fabs((*this)(i, j) - other(i, j)) > tol)
            {
                return false;
            }
        }
    }

    return true;
}

DenseMatrix DenseMatrix::GetRow(size_t start, size_t end) const
{
    const size_t num_rows = end - start;
    DenseMatrix dense(num_rows, cols_);

    GetRow(start, end, dense);

    return dense;
}

void DenseMatrix::GetRow(size_t start, size_t end, DenseMatrix& dense) const
{
    GetSubMatrix(start, 0, end, cols_, dense);
}

DenseMatrix DenseMatrix::GetRow(const std::vector<int>& rows) const
{
    DenseMatrix dense(rows.size(), Cols());

    GetRow(rows, dense);

    return dense;
}

void DenseMatrix::GetRow(const std::vector<int>& rows, DenseMatrix& dense) const
{
    assert(dense.Cols() == Cols());
    assert(dense.Rows() == rows.size());

    const size_t num_rows = rows.size();
    const size_t num_cols = Cols();

    for (size_t i = 0; i < num_rows; ++i)
    {
        const int row = rows[i];

        for (size_t j = 0; j < num_cols; ++j)
        {
            dense(i, j) = (*this)(row, j);
        }
    }

}

void DenseMatrix::SetRow(size_t start, const DenseMatrix& dense)
{
    const size_t end = start + dense.Rows();
    SetSubMatrix(start, 0, end, cols_, dense);
}

DenseMatrix DenseMatrix::GetCol(size_t start, size_t end) const
{
    const size_t num_cols = end - start;
    DenseMatrix dense(rows_, num_cols);

    GetCol(start, end, dense);

    return dense;
}

void DenseMatrix::GetCol(size_t start, size_t end, DenseMatrix& dense) const
{
    GetSubMatrix(0, start, rows_, end, dense);
}

void DenseMatrix::SetCol(size_t start, const DenseMatrix& dense)
{
    const size_t end = start + dense.Cols();
    SetSubMatrix(0, start, rows_, end, dense);
}

DenseMatrix DenseMatrix::GetSubMatrix(size_t start_i, size_t start_j, size_t end_i, size_t end_j) const
{
    const size_t num_rows = end_i - start_i;
    const size_t num_cols = end_j - start_j;

    DenseMatrix dense(num_rows, num_cols);
    GetSubMatrix(start_i, start_j, end_i, end_j, dense);

    return dense;
}

void DenseMatrix::GetSubMatrix(size_t start_i, size_t start_j, size_t end_i, size_t end_j, DenseMatrix& dense) const
{
    assert(start_i >= 0 && start_i < rows_);
    assert(start_j >= 0 && start_j < cols_);
    assert(end_i >= 0 && end_i <= rows_);
    assert(end_j >= 0 && end_j <= cols_);
    assert(end_i >= start_i && end_j >= start_j);

    const size_t num_rows = end_i - start_i;
    const size_t num_cols = end_j - start_j;

    for (size_t j = 0; j < num_cols; ++j)
    {
        for (size_t i = 0; i < num_rows; ++i)
        {
            dense(i, j) = (*this)(i + start_i, j + start_j);
        }
    }
}

void DenseMatrix::SetSubMatrix(size_t start_i, size_t start_j, size_t end_i, size_t end_j, const DenseMatrix& dense)
{
    assert(start_i >= 0 && start_i < rows_);
    assert(start_j >= 0 && start_j < cols_);
    assert(end_i >= 0 && end_i <= rows_);
    assert(end_j >= 0 && end_j <= cols_);
    assert(end_i >= start_i && end_j >= start_j);

    const size_t num_rows = end_i - start_i;
    const size_t num_cols = end_j - start_j;

    for (size_t j = 0; j < num_cols; ++j)
    {
        for (size_t i = 0; i < num_rows; ++i)
        {
            (*this)(i + start_i, j + start_j) = dense(i, j);
        }
    }
}

std::vector<double> DenseMatrix::EigenSolve(DenseMatrix& eigenvectors) const
{
    eigenvectors = *this;

    return eigenvectors.EigenSolve();
}

std::vector<double> DenseMatrix::EigenSolve()
{
    assert(Rows() == Cols());

    const int size = Rows();
    std::vector<double> eigenvalues(size);

    const char* jobz = "V";
    const char* uplo = "U";
    const int* n = &size;
    double* A = data_.data();
    const int* lda = &size;
    double* w = eigenvalues.data();
    int info;

    int lwork = -1;
    int liwork = -1;
    double qwork;
    int qiwork;

    dsyevd_(jobz, uplo, n, A, lda, w, &qwork, &lwork, &qiwork, &liwork, &info);

    lwork = static_cast<int>(qwork);
    liwork = qiwork;

    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);

    dsyevd_(jobz, uplo, n, A, lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);

    assert(info == 0);

    return eigenvalues;
}

std::vector<double> DenseMatrix::SVD(DenseMatrix& U) const
{
    U = *this;

    return U.SVD();
}

std::vector<double> DenseMatrix::SVD()
{
    const int rows = Rows();
    const int cols = Cols();

    std::vector<double> singular_values(std::min(rows, cols));

    if (rows == 0 || cols == 0)
    {
        return singular_values;
    }

    const char* jobu = "O";
    const char* jobvt = "N";
    const int* m = &rows;
    const int* n = &cols;
    double* A = data_.data();
    const int* lda = &rows;
    double* S = singular_values.data();
    double* U = nullptr;
    const int* ldu = &rows;
    double* VT = nullptr;
    const int* ldvt = &cols;
    int info;

    int lwork = -1;
    double qwork;

    dgesvd_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
            &qwork, &lwork, &info);

    lwork = static_cast<int>(qwork);

    std::vector<double> work(lwork);

    dgesvd_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
            work.data(), &lwork, &info);

    assert(info == 0);

    return singular_values;
}

void DenseMatrix::ScaleRows(const std::vector<double>& values)
{
    for (size_t j = 0; j < cols_; ++j)
    {
        for (size_t i = 0; i < rows_; ++i)
        {
            (*this)(i, j) *= values[i];
        }
    }
}

void DenseMatrix::ScaleCols(const std::vector<double>& values)
{
    for (size_t j = 0; j < cols_; ++j)
    {
        const double scale = values[j];

        for (size_t i = 0; i < rows_; ++i)
        {
            (*this)(i, j) *= scale;
        }
    }
}
/*
    void dgeqp3_(const int* m, const int* n, double* A, const int* lda,
                 int* jpvt, double* tau, double* work, const int* lwork, int* info);
    void dorgqr_(const int* m, const int* n, const int* k, double* A,
                 const int* lda, double* tau, double* work, const int* lwork,
                 int* info);
                 */

void DenseMatrix::QR(DenseMatrix& Q) const
{
    Q = *this;
    Q.QR();
}

void DenseMatrix::QR()
{
    const int rows = Rows();
    const int cols = Cols();

    if (rows == 0 || cols == 0)
    {
        return;
    }

    const int mn_min = std::min(rows, cols);

    const int* m = &rows;
    const int* n = &cols;
    double* A = data_.data();
    const int* lda = &rows;
    int info;

    int lwork = -1;
    double qwork;

    dgeqp3_(m, n, A, lda, nullptr, nullptr, &qwork, &lwork, &info);

    lwork = static_cast<int>(qwork);

    std::vector<int> jpvt(mn_min, 0);
    std::vector<double> tau(mn_min);
    std::vector<double> work(lwork);

    dgeqp3_(m, n, A, lda, jpvt.data(), tau.data(), work.data(), &lwork, &info);

    const int* k = &mn_min;

    lwork = -1;
    dorgqr_(m, n, k, A, lda, tau.data(), &qwork, &lwork, &info);
    lwork = static_cast<int>(qwork);

    work.resize(lwork);

    dorgqr_(m, n, k, A, lda, tau.data(), work.data(), &lwork, &info);

    assert(info == 0);
}

std::vector<double> DenseMatrix::GetDiag() const
{
    assert(rows_ == cols_);

    std::vector<double> diag(rows_);

    GetDiag(diag);

    return diag;
}

void DenseMatrix::GetDiag(std::vector<double>& diag) const
{
    assert(rows_ == cols_);
    assert(diag.size() == rows_);

    for (size_t i = 0; i < rows_; ++i)
    {
        diag[i] = (*this)(i, i);
    }
}

} // namespace linalgcpp

