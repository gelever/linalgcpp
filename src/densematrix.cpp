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
    : rows_(0), cols_(0)
{
}

DenseMatrix::DenseMatrix(int size)
    : DenseMatrix(size, size)
{
}

DenseMatrix::DenseMatrix(int rows, int cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0)
{
    assert(rows >= 0);
    assert(cols >= 0);
}

DenseMatrix::DenseMatrix(int rows, int cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data)
{
    assert(rows >= 0);
    assert(cols >= 0);

    assert(data.size() == rows * cols);
}

void DenseMatrix::Print(const std::string& label) const
{
    std::cout << label << "\n";

    const int width = 6;

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            std::cout << std::setw(width) << ((*this)(i, j));

        }

        std::cout << "\n";
    }

    std::cout << "\n";
}

std::vector<double> DenseMatrix::Mult(const std::vector<double>& input) const
{
    std::vector<double> output(rows_);
    Mult(input, output);

    return output;
}

std::vector<double> DenseMatrix::MultAT(const std::vector<double>& input) const
{
    std::vector<double> output(cols_);
    MultAT(input, output);

    return output;
}

void DenseMatrix::Mult(const std::vector<double>& input, std::vector<double>& output) const
{
    assert(input.size() == static_cast<unsigned int>(cols_));
    assert(output.size() == static_cast<unsigned int>(rows_));
    std::fill(begin(output), end(output), 0.0);

    for (int j = 0; j < cols_; ++j)
    {
        for (int i = 0; i < rows_; ++i)
        {
            output[i] += (*this)(i, j) * input[j];
        }
    }
}

void DenseMatrix::MultAT(const std::vector<double>& input, std::vector<double>& output) const
{
    assert(input.size() == static_cast<unsigned int>(rows_));
    assert(output.size() == static_cast<unsigned int>(cols_));

    for (int j = 0; j < cols_; ++j)
    {
        double val = 0.0;

        for (int i = 0; i < rows_; ++i)
        {
            val += (*this)(i, j) * input[i];
        }

        output[j] = val;
    }
}


DenseMatrix DenseMatrix::Mult(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.cols_);
    Mult(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultAT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.cols_);
    MultAT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultBT(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.rows_);
    MultBT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultABT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.rows_);
    MultABT(input, output);

    return output;
}

void DenseMatrix::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(cols_ == input.rows_);
    assert(rows_ == output.rows_);
    assert(input.cols_ == output.cols_);

    bool AT = false;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultAT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(rows_ == input.rows_);
    assert(cols_ == output.rows_);
    assert(input.cols_ == output.cols_);

    bool AT = true;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultBT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(cols_ == input.cols_);
    assert(rows_ == output.rows_);
    assert(input.rows_ == output.cols_);

    bool AT = false;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultABT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(rows_ == input.cols_);
    assert(cols_ == output.rows_);
    assert(input.rows_ == output.cols_);

    bool AT = true;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const
{
    char transA = AT ? 'T' : 'N';
    char transB = BT ? 'T' : 'N';
    int m = AT ? cols_ : rows_;
    int n = BT ? input.rows_ : input.cols_;
    int k = AT ? rows_ : cols_;

    double alpha = 1.0;
    const double* A = data_.data();
    int lda = rows_;
    const double* B = input.data_.data();
    int ldb = input.rows_;
    double beta = 0.0;
    double* c = output.data_.data();
    int ldc = output.rows_;

    dgemm_(&transA, &transB, &m, &n, &k,
           &alpha, A, &lda, B, &ldb,
           &beta, c, &ldc);
}

DenseMatrix& DenseMatrix::operator-=(const DenseMatrix& other)
{
    assert(rows_ == other.rows_);
    assert(cols_ == other.cols_);

    const int nnz = rows_ * cols_;

    for (int i = 0; i < nnz; ++i)
    {
        data_[i] -= other.data_[i];
    }

    return *this;
}

DenseMatrix& DenseMatrix::operator+=(const DenseMatrix& other)
{
    assert(rows_ == other.rows_);
    assert(cols_ == other.cols_);

    const int nnz = rows_ * cols_;

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

