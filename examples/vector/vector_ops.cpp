#include "../include/ex_utilities.hpp"

Vector read_vector(const std::string& filename)
{
    std::vector<double> data = linalgcpp::ReadText(filename);

    return Vector(data);
}

double inner_product(const Vector& v1, const Vector& v2)
{
    assert(v1.size() == v2.size());

    int size = v1.size();
    double sum = 0.0;

    for (int i = 0; i < size; ++i)
    {
        sum += v1[i] * v2[i];
    }

    return sum;
}

double l2_norm(const Vector& vector)
{
    return std::sqrt(inner_product(vector, vector));
}

int main()
{
    Vector v;
    Vector v2(4);

    v2.Print("vect");

    int dim = 6;

    Vector zeros(dim, 0.0);
    Vector ones(dim, 1.0);
    Vector twos(dim, 2.0);

    ones.Print("ones");

    Vector vect_from_file = read_vector("vect.text");
    vect_from_file.Print("From file");

    double ip = inner_product(twos, ones);
    std::cout << "Inner product: " << ip << "\n";

    double norm = l2_norm(twos);
    std::cout << "Norm: " << norm << "\n";

    twos /= norm;
    twos.Print("twos normalized");

    double two_ip = inner_product(twos, twos);
    std::cout << "Twos Inner product: " << two_ip << "\n";
    twos += ones;

    Vector three = ones + twos;

}
