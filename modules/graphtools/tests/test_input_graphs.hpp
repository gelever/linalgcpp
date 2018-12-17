#pragma once


template <typename T>
struct InputGraph
{
    linalgcpp::SparseMatrix<T> vertex_edge;
    std::vector<int> partition;
    int max_procs;
};

// 
// *--*\ /*--*
// |  | * |  |
// *--*/ \*--*
// 
// 
template <typename T>
InputGraph<T> TestGraph0()
{
    linalgcpp::CooMatrix<T> edge_vertex(12, 9);

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
    linalgcpp::CooMatrix<T> edge_vertex(10, 9);

    edge_vertex.Add(0, 0, 1.0);
    edge_vertex.Add(0, 1, 1.0);

    edge_vertex.Add(1, 0, 1.0);
    edge_vertex.Add(1, 2, 1.0);

    edge_vertex.Add(2, 1, 1.0);
    edge_vertex.Add(2, 2, 1.0);

    edge_vertex.Add(3, 3, 1.0);
    edge_vertex.Add(3, 4, 1.0);

    edge_vertex.Add(4, 3, 1.0);
    edge_vertex.Add(4, 5, 1.0);

    edge_vertex.Add(5, 4, 1.0);
    edge_vertex.Add(5, 5, 1.0);

    edge_vertex.Add(6, 6, 1.0);
    edge_vertex.Add(6, 7, 1.0);

    edge_vertex.Add(7, 6, 1.0);
    edge_vertex.Add(7, 8, 1.0);

    edge_vertex.Add(8, 7, 1.0);
    edge_vertex.Add(8, 8, 1.0);

    edge_vertex.Add(9, 2, 1.0);
    edge_vertex.Add(9, 3, 1.0);
    edge_vertex.Add(9, 6, 1.0);

    std::vector<int> part{0, 0, 0,
                          1, 1, 1,
                          2, 2, 2};
    int max_procs = 3;

    return {edge_vertex.ToSparse().Transpose(), part, max_procs};
}

template <typename T>
std::vector<InputGraph<T>>
GetAllGraphs()
{
    return
    {
        TestGraph0<T>(),
        TestGraph1<T>()
    };
}

