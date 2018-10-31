#include "../include/ex_utilities.hpp"

int main()
{
    Vector v;
    Vector v2(4);

    v.Print("empty vect");
    v2.Print("size 4 vect");

    int dim = 6;

    Vector zeros(dim, 0.0);
    Vector ones(dim, 1.0);
    Vector twos(dim, 2.0);

    std::cout << "Ones size: " << ones.size() << "\n";
    ones.Print("ones");

    zeros[0] = 2.0;
    zeros[1] *= 8.0;

    std::cout << "Modified 0th element: " << zeros[0] << " ";
    std::cout << "1st element: " << zeros[1] << "\n";

    Vector vect_from_file(linalgcpp::ReadText("data/vect4.txt"));
    vect_from_file.Print("From file");

    double ip = twos.Mult(ones);
    std::cout << "Inner product: " << ip << "\n";

    double norm = twos.L2Norm();
    std::cout << "Norm: " << norm << "\n";

    Vector twos_normalized = twos;
    twos_normalized /= norm;
    twos_normalized.Print("twos normalized");

    double two_ip = twos.Mult(twos);
    std::cout << "Twos Inner product: " << two_ip << "\n";

    Vector threes = twos;
    threes += ones;
    threes.Print("1 + 2");

    Vector fours = twos + twos;
    fours.Print("2 + 2");
}
