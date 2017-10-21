#include "densematrix.hpp"

extern "C"
{
    void dgemm_(const char* transA, const char* transB,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb, const double* beta,
                double* C, const int* ldc);
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

DenseMatrix::DenseMatrix(DenseMatrix&& other)
{
    Swap(*this, other);
}

void Swap(DenseMatrix& lhs, DenseMatrix& rhs)
{
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.cols_, rhs.cols_);
    std::swap(lhs.data_, rhs.data_);
}

void DenseMatrix::Print(const std::string& label) const
{
    std::cout << label << "\n";

    const int width = 6;

    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            std::cout << std::setw(width) << (*this)(i, j);
        }

        std::cout << "\n";
    }

    std::cout << "\n";
}

void DenseMatrix::Mult(const Vector<double>& input, Vector<double>& output) const
{
    Mult<double, double>(input, output);
}

void DenseMatrix::MultAT(const Vector<double>& input, Vector<double>& output) const
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

} // namespace linalgcpp

