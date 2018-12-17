#pragma once


template <typename T>
struct InputGraph
{
    linalgcpp::SparseMatrix<T> vertex_edge;
    std::vector<int> partition;
    int max_procs;
};

template <typename T>
void AddEdge(linalgcpp::CooMatrix<T>& coo, const std::vector<int>& vertices)
{
    int edge_num = std::get<0>(coo.FindSize());

    for (auto&& vertex : vertices)
    {
        coo.Add(edge_num, vertex, (T)1.0);
    }
}

// 
// *--*\ /*--*
// |  | * |  |
// *--*/ \*--*
// 
// 
template <typename T>
InputGraph<T> TestGraph0()
{
    linalgcpp::CooMatrix<T> edge_vertex;

    AddEdge(edge_vertex, {0, 1});
    AddEdge(edge_vertex, {0, 2});
    AddEdge(edge_vertex, {1, 3});
    AddEdge(edge_vertex, {1, 4});
    AddEdge(edge_vertex, {2, 3});
    AddEdge(edge_vertex, {3, 4});
    AddEdge(edge_vertex, {4, 5});
    AddEdge(edge_vertex, {4, 7});
    AddEdge(edge_vertex, {5, 6});
    AddEdge(edge_vertex, {5, 7});
    AddEdge(edge_vertex, {6, 8});
    AddEdge(edge_vertex, {7, 8});

    std::vector<int> part{0, 0, 0, 0, 0,
                          1, 1, 1, 1};

    int max_procs = 2;

    return {edge_vertex.ToSparse().Transpose(), part, max_procs};
}

//
//
//  *      *
//  |\    /|
//  *-*--*-*
//     \/  // 3 vertex edge in middle
//     *
//    / \
//   *---*
//
template <typename T>
InputGraph<T> TestGraph1()
{
    linalgcpp::CooMatrix<T> edge_vertex;

    AddEdge(edge_vertex, {0, 1});
    AddEdge(edge_vertex, {0, 2});
    AddEdge(edge_vertex, {1, 2});
    AddEdge(edge_vertex, {3, 4});
    AddEdge(edge_vertex, {3, 5});
    AddEdge(edge_vertex, {4, 5});
    AddEdge(edge_vertex, {6, 7});
    AddEdge(edge_vertex, {6, 8});
    AddEdge(edge_vertex, {7, 8});
    AddEdge(edge_vertex, {2, 3, 6});

    std::vector<int> part{0, 0, 0,
                          1, 1, 1,
                          2, 2, 2};
    int max_procs = 3;

    return {edge_vertex.ToSparse().Transpose(), part, max_procs};
}

template <typename T>
std::vector<InputGraph<T>> GetAllGraphs()
{
    return
    {
        TestGraph0<T>(),
        TestGraph1<T>()
    };
}

