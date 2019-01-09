#include "graphtools.hpp"
#include "test_input_graphs.hpp"

using namespace linalgcpp;

template <typename T>
void CreateGraph(const MpiSession& mpi, const InputGraph<T>& graph_info)
{
    const auto& vertex_edge = graph_info.vertex_edge;
    const auto& part = graph_info.partition;
    const auto& max_procs = graph_info.max_procs;

    if (mpi.num_procs > max_procs)
    {
        return;
    }

    Graph<T> graph(mpi.comm, vertex_edge, part);
    GraphTopology<T> gt(graph);

    for (int i = 0; i < mpi.num_procs; ++i)
    {
        if (mpi.myid == i)
        {
            std::cout << "Processor: " << i << "\n";
            std::cout << "Type: " << typeid(T).name() << "\n";
            graph.vertex_edge_local_.ToDense().Print("VE:", std::cout, 4, 0);
            gt.face_agg_local_.ToDense().Print("face_agg :", std::cout, 4, 0);
            gt.face_edge_.GetDiag().ToDense().Print("face_edge diag:", std::cout, 4, 0);
            gt.face_edge_.GetOffd().ToDense().Print("face_edge offd:", std::cout, 4, 0);
            gt.face_true_face_.GetDiag().ToDense().Print("face_true_face_ diag:", std::cout, 4, 0);
            gt.face_true_face_.GetOffd().ToDense().Print("face_true_face_ offd:", std::cout, 4, 0);
            std::cout.flush();
        }

        MPI_Barrier(mpi.comm);
    }
}

template <typename T>
void TestAllGraphs(const MpiSession& mpi)
{
    for (const auto& graph_info_i : GetAllGraphs<T>())
    {
        CreateGraph(mpi, graph_info_i);
    }
}

int main(int argc, char** argv)
{
    MpiSession mpi(argc, argv);

    // Test both types of relationships
    TestAllGraphs<int>(mpi);
    TestAllGraphs<double>(mpi);
}
