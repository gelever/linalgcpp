#include "graphtools.hpp"
#include "test_input_graphs.hpp"

using namespace linalgcpp;

int main(int argc, char** argv)
{
    MpiSession mpi(argc, argv);

    auto graph_input = TestGraph0<double>();

    auto ve = graph_input.vertex_edge;
    auto ev = ve.Transpose();
    auto vv = ve.Mult(ev);

    vv.ToDense().Print("vv:", std::cout, 4, 0);

    ParMatrix A(mpi.comm, vv);
    A.GetDiag().ToDense().Print("A:", std::cout, 4, 0);

    ParMatrix PT = ParPartition(A, 2);
    PT.GetDiag().ToDense().Print("P:", std::cout, 4, 0);
}
