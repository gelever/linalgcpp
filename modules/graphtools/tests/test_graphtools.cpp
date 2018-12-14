#include "graphtools.hpp"

using namespace linalgcpp;

template <typename T>
SparseMatrix<T> MakeVertexEdge()
{
    CooMatrix<T> edge_vertex(12, 9);

    edge_vertex.Add(0, 0, 1.0);
    edge_vertex.Add(0, 1, 1.0);

    edge_vertex.Add(1, 0, 1.0);
    edge_vertex.Add(1, 2, 1.0);

    edge_vertex.Add(2, 1, 1.0);
    edge_vertex.Add(2, 3, 1.0);

    edge_vertex.Add(3, 1, 1.0);
    edge_vertex.Add(3, 4, 1.0);

    edge_vertex.Add(4, 2, 1.0);
    edge_vertex.Add(4, 3, 1.0);

    edge_vertex.Add(5, 3, 1.0);
    edge_vertex.Add(5, 4, 1.0);

    edge_vertex.Add(6, 4, 1.0);
    edge_vertex.Add(6, 5, 1.0);

    edge_vertex.Add(7, 4, 1.0);
    edge_vertex.Add(7, 7, 1.0);

    edge_vertex.Add(8, 5, 1.0);
    edge_vertex.Add(8, 6, 1.0);

    edge_vertex.Add(9, 5, 1.0);
    edge_vertex.Add(9, 7, 1.0);

    edge_vertex.Add(10, 6, 1.0);
    edge_vertex.Add(10, 8, 1.0);

    edge_vertex.Add(11, 7, 1.0);
    edge_vertex.Add(11, 8, 1.0);

    return edge_vertex.ToSparse().Transpose();
}

template <typename T>
void CreateGraph(const MpiSession& mpi, const std::vector<int>& part)
{
    Graph<T> graph(mpi.comm, MakeVertexEdge<T>(), part);
    GraphTopology<T> gt(graph);
    ParMatrix mis_dof = MakeMISDof(graph, gt);

    for (int i = 0; i < mpi.num_procs; ++i)
    {
        if (mpi.myid == i)
        {
            std::cout << "Processor: " << i << "\n";
            std::cout << "Type: " << typeid(T).name() << "\n";
            graph.vertex_edge_local_.ToDense().Print("VE:", std::cout, 4, 0);
            mis_dof.GetDiag().ToDense().Print("MIS_dof diag:", std::cout, 4, 0);
            mis_dof.GetOffd().ToDense().Print("MIS_dof offd:", std::cout, 4, 0);
            std::cout.flush();
        }

        MPI_Barrier(mpi.comm);
    }
}

int main(int argc, char** argv)
{
    MpiSession mpi(argc, argv);

    linalgcpp_parverify(mpi.comm, mpi.num_procs <= 2,
                        "Small test designed for at most two processors!");

    std::vector<int> part{0, 0, 0, 0, 0,
                          1, 1, 1, 1};

    // Type based relationship
    {
        CreateGraph<int>(mpi, part);
        CreateGraph<double>(mpi, part);
    }

    // Using alias to reduce verbosity
    {
        using Graph = linalgcpp::Graph<int>;
        using GraphTopology = linalgcpp::GraphTopology<int>;

        Graph graph(mpi.comm, MakeVertexEdge<int>(), part);
        GraphTopology gt(graph);
    }
}
