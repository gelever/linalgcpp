#include "../include/ex_utilities.hpp"

Vector parse_vector(const std::string& filename)
{
    std::vector<double> data = linalgcpp::ReadText(filename);

    return Vector(std::move(data));
}

// Vector Constructors
void constructors()
{
    // Empty
    Vector v_0;
    Vector v_2(2);

    // Filled
    int dim = 4;

    Vector zeros(dim, 0.0);
    Vector ones(dim, 1.0);
    Vector three_halfs(dim, 3.0 / 2.0);

    // From File
    Vector from_file = parse_vector("data/vect2.txt");

    // Print all examples
    v_0.Print("v_0:");
    v_2.Print("v_2:");

    zeros.Print("zeros:");
    ones.Print("ones:");
    three_halfs.Print("twos:");

    from_file.Print("from_file:");
}

double inner_product(const Vector& lhs, const Vector& rhs)
{
    int size = lhs.size();
    assert(rhs.size() == size);

    double sum = 0.0;

    std::cout << "Inner Product: ";

    for (int i = 0; i < size; ++i)
    {
        std::cout << "(" << lhs[i] << " * " << rhs[i] << ")";
        if (i != size - 1) std::cout << " + ";

        sum += lhs[i] * rhs[i];
    }

    std::cout << "\n";

    return sum;
}

// Vector Operations
void ops()
{
    int dim = 4;

    Vector v1 = parse_vector("data/vect4.txt");
    Vector v2 = parse_vector("data/vect4.2.txt");

    v1.Print("Vector 1:");
    v2.Print("Vector 2:");

    std::cout << "Vector 1 size: " << v1.size() << "\n";
    std::cout << "Vector 2 size: " << v2.size() << "\n";
    std::cout << "\nv1[0]: " << v1[0] << "\n";
    std::cout << "v1[1]: " << v1[1] << "\n";

    v1[0] = 3.14;
    std::cout << "v1[0]: " << v1[0] << "\n\n";

    Vector v1_plus_v2 = v1 + v2;
    Vector v1_sub_v2 = v1 - v2;

    v1_plus_v2.Print("v1 + v2:");
    v1_sub_v2.Print("v1 - v2:");

    Vector v1_x_10 = 10.0 * v1;
    v1_x_10.Print("10.0 * v1");

    double v1_dot_v2 = inner_product(v1, v2);
    std::cout << "v1 * v2: " << v1_dot_v2 << "\n";

    double v1_norm = std::sqrt(inner_product(v1, v1));
    double v2_norm = std::sqrt(inner_product(v2, v2));

    std::cout << "|| v1 ||: " << v1_norm << "\n";
    std::cout << "|| v2 ||: " << v2_norm << "\n";
}

int main(int argc, char** argv)
{
    constructors();
    ops();

    return EXIT_SUCCESS;
}
