#include "../include/ex_utilities.hpp"

void constructors()
{
    DenseMatrix d_0;
    DenseMatrix d_2_2(2);
    DenseMatrix d_2_3(2, 3);

    d_0.Print("d 0x0");
    d_2_2.Print("d 2x2");
    d_2_3.Print("d 2x3");

    DenseMatrix from_file = parse_dense("data/mat2.txt");
    from_file.Print("From File:");
}

void matvec_ops()
{
    DenseMatrix m(4, 2);

    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(2, 0) = 3.0;
    m(3, 1) = 4.0;

    m.Print("m:");

    DenseMatrix m_T = m.Transpose();
    m_T.Print("m_T:");

    Vector b_2(2);
    b_2[0] = -1.0; b_2[1] = 3.0;

    Vector b_4(4);
    b_4[0] = -1.5; b_4[1] = 2.5;
    b_4[2] = -1.5; b_4[3] = 3.5;

    Vector mb = m.Mult(b_2);
    Vector m_Tb = m.MultAT(b_4);

    mb.Print("m * b_2");
    m_Tb.Print("m^T * b_4");
}

void access_ops()
{
    int dim = 4;

    DenseMatrix d_0 = parse_dense("data/mat4.txt");
    DenseMatrix d_1 = parse_dense("data/mat4.2.txt");

    d_0.Print("d_0", std::cout, 6, 1);
    d_1.Print("d_1", std::cout, 6, 1);

    std::cout << "d_0 size: " << d_0.Rows() << " x " << d_0.Cols() << "\n";
    std::cout << "d_1 size: " << d_1.Rows() << " x " << d_1.Cols() << "\n\n";

    std::cout << "d_0 (1, 2): " << d_0(1, 2) << "\n";
    std::cout << "d_1 (2, 3): " << d_1(2, 2) << "\n\n";

    DenseMatrix d0_plus_d1 = d_0 + d_1;
    DenseMatrix d0_sub_d1 = d_0 - d_1;

    d0_plus_d1.Print("d0 + d1");
    d0_sub_d1.Print("d0 - d1");

    DenseMatrix d1_x_10 = 10.0 * d_1;
    d1_x_10.Print("10.0 * d0");

    d1_x_10 /= 4.0;
    d1_x_10.Print("(10.0 * d0) / 4.0");
}

void matmat_ops()
{
    DenseMatrix m(4, 2);

    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(2, 0) = 3.0; m(3, 1) = 4.0;

    DenseMatrix m_T = m.Transpose();

    m.Print("m:");
    m_T.Print("m_T:");

    DenseMatrix m_T_m = m_T.Mult(m);
    m_T_m.Print("m^T * m");

    DenseMatrix m_m_T = m.Mult(m_T);
    m_m_T.Print("m * m^T");
}

int main()
{
    constructors();

    access_ops();
    matvec_ops();
    matmat_ops();
}
