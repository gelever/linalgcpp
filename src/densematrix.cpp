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
    : Operator(rows, cols), data_(rows * cols, 0.0)
{
}

DenseMatrix::DenseMatrix(size_t rows, size_t cols, const std::vector<double>& data)
    : Operator(rows, cols), data_(data)
{
    assert(data.size() == rows * cols);
}

void Swap(DenseMatrix& lhs, DenseMatrix& rhs)
{
    Swap(static_cast<Operator&>(lhs), static_cast<Operator&>(rhs));
    std::swap(lhs.data_, rhs.data_);
}

void DenseMatrix::Print(const std::string& label) const
{
    std::cout << label << "\n";

    const int width = 6;

    for (size_t i = 0; i < Rows(); ++i)
    {
        for (size_t j = 0; j < Cols(); ++j)
        {
            std::cout << std::setw(width) << ((*this)(i, j));

        }

        std::cout << "\n";
    }

    std::cout << "\n";
}

DenseMatrix DenseMatrix::Mult(const DenseMatrix& input) const
{
    DenseMatrix output(Rows(), input.Cols());
    Mult(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultAT(const DenseMatrix& input) const
{
    DenseMatrix output(Cols(), input.Cols());
    MultAT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultBT(const DenseMatrix& input) const
{
    DenseMatrix output(Rows(), input.Rows());
    MultBT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultABT(const DenseMatrix& input) const
{
    DenseMatrix output(Cols(), input.Rows());
    MultABT(input, output);

    return output;
}

void DenseMatrix::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(Cols() == input.Rows());
    assert(Rows() == output.Rows());
    assert(input.Cols() == output.Cols());

    bool AT = false;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultAT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(Rows() == input.Rows());
    assert(Cols() == output.Rows());
    assert(input.Cols() == output.Cols());

    bool AT = true;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultBT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(Cols() == input.Cols());
    assert(Rows() == output.Rows());
    assert(input.Rows() == output.Cols());

    bool AT = false;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultABT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(Rows() == input.Cols());
    assert(Cols() == output.Rows());
    assert(input.Rows() == output.Cols());

    bool AT = true;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const
{
    char transA = AT ? 'T' : 'N';
    char transB = BT ? 'T' : 'N';
    int m = AT ? Cols() : Rows();
    int n = BT ? input.Rows() : input.Cols();
    int k = AT ? Rows() : Cols();

    double alpha = 1.0;
    const double* A = data_.data();
    int lda = Rows();
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
    assert(Rows() == other.Rows());
    assert(Cols() == other.Cols());

    const size_t nnz = Rows() * Cols();

    for (size_t i = 0; i < nnz; ++i)
    {
        data_[i] -= other.data_[i];
    }

    return *this;
}

DenseMatrix& DenseMatrix::operator+=(const DenseMatrix& other)
{
    assert(Rows() == other.Rows());
    assert(Cols() == other.Cols());

    const int nnz = Rows() * Cols();

    for (int i = 0; i < nnz; ++i)
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

