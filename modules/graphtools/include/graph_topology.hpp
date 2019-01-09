/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

    @brief GraphTopology class
*/

#ifndef GRAPHTOPOLOGY_HPP
#define GRAPHTOPOLOGY_HPP

#include "graph.hpp"
#include "graph_mis.hpp"
#include "graph_utilities.hpp"

namespace linalgcpp
{

/**
   @brief Class to represent the topology of a graph as it is coarsened.

   Mostly a container for a bunch of topology tables.
*/

template <typename T = double>
class GraphTopology
{
public:
    /** @brief Default Constructor */
    GraphTopology() = default;

    /**
       @brief Build agglomerated topology relation tables of a given graph

       @param graph Distrubted graph information
    */
    GraphTopology(const Graph<T>& graph);

    /**
        @brief Build agglomerated topology relation tables of the coarse level
               graph in a given GraphTopology object

        @param finer_graph_topology finer level graph topology
        @param coarsening_factor intended number of vertices in an aggregate
    */
    GraphTopology(const GraphTopology& fine_topology, double coarsening_factor);

    /**
       @brief Build agglomerated topology relation tables of a given graph
              Using already distributed information

       @param vertex_edge local vertex to edge relationship
       @param partition local vertex partition
       @param edge_true_edge edge to true edge relationship
    */
    GraphTopology(const SparseMatrix<T>& vertex_edge,
                  const std::vector<int>& partition,
                  ParMatrix edge_true_edge);

    /** @brief Default Destructor */
    ~GraphTopology() noexcept = default;

    /** @brief Copy Constructor */
    GraphTopology(const GraphTopology& other) noexcept;

    /** @brief Move Constructor */
    GraphTopology(GraphTopology&& other) noexcept;

    /** @brief Assignment Operator */
    GraphTopology& operator=(GraphTopology other) noexcept;

    /** @brief Swap two topologies */
    template <typename U>
    friend void swap(GraphTopology<U>& lhs, GraphTopology<U>& rhs) noexcept;

    int NumAggs() const { return agg_vertex_local_.Rows(); }
    int NumVertices() const { return agg_vertex_local_.Cols(); }
    int NumEdges() const { return agg_edge_local_.Cols(); }
    int NumFaces() const { return agg_face_local_.Cols(); }

    int GlobalNumAggs() const { return agg_ext_vertex_.GlobalRows(); }
    int GlobalNumVertices() const { return agg_ext_vertex_.GlobalCols(); }
    int GlobalNumEdges() const { return face_edge_.GlobalCols(); }
    int GlobalNumFaces() const { return face_edge_.GlobalRows(); }

    // Local topology
    SparseMatrix<T> agg_vertex_local_; // Aggregate to vertex, not exteneded
    SparseMatrix<T> agg_edge_local_;   // Aggregate to edge, not extended
    SparseMatrix<T> face_agg_local_;   // Face to aggregate
    SparseMatrix<T> agg_face_local_;   // Aggregate to face

    // Global topology
    ParMatrix face_true_face_; // Face to true face
    ParMatrix face_edge_;      // Face to false edge
    ParMatrix agg_ext_vertex_; // Aggregate to extended vertex
    ParMatrix agg_ext_edge_;   // Aggregate to extended edge
    ParMatrix edge_true_edge_;   // Edge to true edge

private:
    void Init(const SparseMatrix<T>& vertex_edge,
              const std::vector<int>& partition,
              ParMatrix edge_true_edge);

    SparseMatrix<double> MakeFaceIntAgg(const ParMatrix& agg_agg);

    SparseMatrix<double> MakeFaceEdge(const ParMatrix& agg_agg,
                              const ParMatrix& edge_edge,
                              const SparseMatrix<T>& agg_edge_ext,
                              const SparseMatrix<double>& face_int_agg_edge);

    SparseMatrix<T> ExtendFaceAgg(const ParMatrix& agg_agg,
                               const SparseMatrix<double>& face_int_agg);

};

template <typename T>
GraphTopology<T>::GraphTopology(const Graph<T>& graph)
{
    const auto& vertex_edge = graph.vertex_edge_local_;
    const auto& part = graph.part_local_;

    Init(vertex_edge, part,  graph.edge_true_edge_);
}

template <typename T>
GraphTopology<T>::GraphTopology(const GraphTopology<T>& fine_topology,
                                double coarsening_factor)
{
    const auto& vertex_edge = fine_topology.agg_face_local_;
    const auto& part = PartitionAAT(vertex_edge, coarsening_factor, 1.2);

    Init(vertex_edge, part, fine_topology.face_true_face_);
}

template <typename T>
GraphTopology<T>::GraphTopology(const SparseMatrix<T>& vertex_edge,
                                const std::vector<int>& partition,
                                ParMatrix edge_true_edge)
{
    const auto true_edge_edge = edge_true_edge.Transpose();
    const auto edge_edge = edge_true_edge.Mult(true_edge_edge);

    Init(vertex_edge, partition, std::move(edge_true_edge));
}

template <typename T>
SparseMatrix<T> RemoveEmptyRows(const SparseMatrix<T>& mat)
{
    CooMatrix<T> coo;

    int num_rows = mat.Rows();

    std::vector<int> indptr(1, 0);
    std::vector<int> indices = mat.GetIndices();
    std::vector<T> data = mat.GetData();

    for (int i = 0; i < num_rows; ++i)
    {
        int row_size = mat.RowSize(i);

        if (mat.RowSize(i) > 0)
        {
            indptr.push_back(indptr.back() + row_size);
        }
    }

    linalgcpp_verify(indptr.back() == indices.size());

    int new_rows = indptr.size() - 1;
    int new_cols = mat.Cols();

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                           new_rows, new_cols);
}

template <typename T>
bool has_empty_row(const SparseMatrix<T>& mat)
{
    for (int i = 0; i < mat.Rows(); ++i)
    {
        if (mat.RowSize(i) == 0)
        {
            return true;
        }
    }
    return false;
}

inline
bool has_empty_row(const ParMatrix& mat)
{
    for (int i = 0; i < mat.Rows(); ++i)
    {
        if (mat.RowSize(i) == 0)
        {
            return true;
        }
    }
    return false;
}


template <typename T>
void GraphTopology<T>::Init(const SparseMatrix<T>& vertex_edge,
                            const std::vector<int>& partition,
                            ParMatrix edge_true_edge)
{
    auto edge_edge = edge_true_edge.Mult(edge_true_edge.Transpose());
    edge_true_edge_ = std::move(edge_true_edge);

    MPI_Comm comm = edge_true_edge_.GetComm();

    agg_vertex_local_ = MakeSetEntity<T>(partition);

    int num_vertices = vertex_edge.Rows();
    int num_edges = vertex_edge.Cols();
    int num_aggs = agg_vertex_local_.Rows();

    auto starts = linalgcpp::GenerateOffsets(comm, {num_vertices, num_edges, num_aggs});

    const auto& vertex_starts = starts[0];
    const auto& edge_starts = starts[1];
    const auto& agg_starts = starts[2];

    ParMatrix vertex_edge_d(comm, vertex_starts, edge_starts, Duplicate<T, double>(vertex_edge));
    ParMatrix vertex_true_edge = RemoveLargeEntries(vertex_edge_d.Mult(edge_true_edge_));
    ParMatrix true_edge_vertex = vertex_true_edge.Transpose();
    ParMatrix true_edge_edge = edge_true_edge_.Transpose();

    ParMatrix agg_vertex_d(comm, Duplicate<T, double>(agg_vertex_local_));
    ParMatrix agg_true_edge_d = agg_vertex_d.Mult(vertex_true_edge);

    SparseMatrix<T> agg_edge_ext = agg_vertex_local_.Mult(vertex_edge);
    agg_edge_ext.SortIndices();

    //agg_edge_local_ = RemoveLargeEntries(agg_edge_ext, 1.5);
    //auto edge_vertex = vertex_edge.Transpose();
    //agg_edge_local_ = RemoveLowDegree(agg_edge_ext, edge_vertex);
    agg_edge_local_ = Duplicate<double, T>(RemoveLowDegree(agg_true_edge_d, true_edge_vertex).Mult(true_edge_edge).GetDiag());

    agg_edge_local_ = 1.0;

    SparseMatrix<double> edge_ext_agg = agg_edge_ext.template Transpose<double>();

    //ParMatrix edge_agg_d(comm, edge_starts, agg_starts, std::move(edge_ext_agg));
    ParMatrix edge_agg_d(comm, edge_starts, agg_starts, (edge_ext_agg));
    ParMatrix agg_edge_d = edge_agg_d.Transpose();

    ParMatrix edge_agg_ext = edge_edge.Mult(edge_agg_d);
    ParMatrix agg_agg = agg_edge_d.Mult(edge_agg_ext);

    const auto& edge_v_edge = true_edge_vertex.Mult(vertex_true_edge);
    const auto& agg_r = MakeExtPermutation(agg_agg);
    const auto& edge_r = MakeExtPermutation(edge_v_edge);
    const auto& edge_r_T = edge_r.Transpose();
    const auto& agg_edge = agg_edge_d.Mult(edge_true_edge_);

    // HYPRE kindly adds explicit zeros on diagonal
    auto agg_edge_r = RemoveLargeEntries(agg_r.Mult(agg_edge).Mult(edge_r_T), 0.5);

    auto local_agg_edge = agg_edge_r.GetDiag();
    auto local_edge_agg = agg_edge_r.GetDiag().Transpose();

    auto mis = GenerateMIS<double>(local_agg_edge, local_edge_agg);
    auto mis_dof = MakeSetEntity<double>(mis);

    auto mis_agg_f = mis_dof.Mult(local_edge_agg);
    auto agg_mis_f = mis_agg_f.Transpose();

    auto mis_agg_ur = ParMatrix(comm, mis_agg_f).Mult(agg_r);
    auto face_mis = MakeFaceMIS(mis_agg_f);

    //auto face_agg = face_mis.Mult(mis_agg_f);
    auto face_dof = face_mis.Mult(mis_dof);

    ParMatrix face_dof_par(comm, std::move(face_dof));
    //ParMatrix face_agg_par(comm, std::move(face_agg));

    ParMatrix face_dof_ur = RemoveLargeEntries(face_dof_par.Mult(edge_r), 0.5);
    //ParMatrix face_agg_ur = RemoveLargeEntries(face_agg_par.Mult(agg_r), 0.5);

    face_dof_par = 1.0;
    //face_agg_par = 1.0;

    auto face_edge = RemoveEmptyRows(face_dof_ur.Mult(true_edge_edge).GetDiag());
    auto face_agg_local = face_edge.Mult(edge_ext_agg);
    face_edge_ = ParMatrix(comm, std::move(face_edge));

    auto face_face = (face_edge_.Mult(edge_edge).Mult(face_edge_.Transpose()));
    face_true_face_ = MakeEntityTrueEntity(face_face);
    face_true_face_ = 1.0;

    //face_agg_local_ = RemoveEmptyRows(Duplicate<double, T>(face_agg_ur.GetDiag()));
    face_agg_local_ = Duplicate<double, T>(face_agg_local);
    face_agg_local_ = 1.0;

    agg_face_local_ = face_agg_local_.Transpose();

    agg_ext_vertex_ = agg_edge.Mult(true_edge_vertex);
    agg_ext_vertex_ = 1.0;
    vertex_true_edge = 1.0;

    //ParMatrix agg_ext_edge_ext = agg_ext_vertex_.Mult(vertex_true_edge);
    //agg_ext_edge_ = RemoveLowDegree(agg_ext_edge_ext, true_edge_vertex);

    agg_ext_edge_ = agg_true_edge_d;
}

template <typename T>
GraphTopology<T>::GraphTopology(const GraphTopology<T>& other) noexcept
    : agg_vertex_local_(other.agg_vertex_local_),
      agg_edge_local_(other.agg_edge_local_),
      face_agg_local_(other.face_agg_local_),
      agg_face_local_(other.agg_face_local_),
      face_true_face_(other.face_true_face_),
      face_edge_(other.face_edge_),
      agg_ext_vertex_(other.agg_ext_vertex_),
      agg_ext_edge_(other.agg_ext_edge_),
      edge_true_edge_(other.edge_true_edge_)
{

}

template <typename T>
GraphTopology<T>::GraphTopology(GraphTopology<T>&& other) noexcept
{
    swap(*this, other);
}

template <typename T>
GraphTopology<T>& GraphTopology<T>::operator=(GraphTopology<T> other) noexcept
{
    swap(*this, other);

    return *this;
}

template <typename T>
void swap(GraphTopology<T>& lhs, GraphTopology<T>& rhs) noexcept
{
    swap(lhs.agg_vertex_local_, rhs.agg_vertex_local_);
    swap(lhs.agg_edge_local_, rhs.agg_edge_local_);
    swap(lhs.face_agg_local_, rhs.face_agg_local_);
    swap(lhs.agg_face_local_, rhs.agg_face_local_);

    swap(lhs.face_true_face_, rhs.face_true_face_);
    swap(lhs.face_edge_, rhs.face_edge_);
    swap(lhs.agg_ext_vertex_, rhs.agg_ext_vertex_);
    swap(lhs.agg_ext_edge_, rhs.agg_ext_edge_);
    swap(lhs.edge_true_edge_, rhs.edge_true_edge_);
}

template <typename T>
SparseMatrix<double> GraphTopology<T>::MakeFaceIntAgg(const ParMatrix& agg_agg)
{
    const auto& agg_agg_diag = agg_agg.GetDiag();

    int num_aggs = agg_agg_diag.Rows();
    int num_faces = agg_agg_diag.nnz() - agg_agg_diag.Rows();

    assert(num_faces % 2 == 0);
    num_faces /= 2;

    std::vector<int> indptr(num_faces + 1);
    std::vector<int> indices(num_faces * 2);
    std::vector<double> data(num_faces * 2, 1);

    indptr[0] = 0;

    const auto& agg_indptr = agg_agg_diag.GetIndptr();
    const auto& agg_indices = agg_agg_diag.GetIndices();
    int rows = agg_agg_diag.Rows();
    int count = 0;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = agg_indptr[i]; j < agg_indptr[i + 1]; ++j)
        {
            if (agg_indices[j] > i)
            {
                indices[count * 2] = i;
                indices[count * 2 + 1] = agg_indices[j];

                count++;

                indptr[count] = count * 2;
            }
        }
    }

    assert(count == num_faces);

    return SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_aggs);
}

template <typename T>
SparseMatrix<double> GraphTopology<T>::MakeFaceEdge(const ParMatrix& agg_agg,
                                         const ParMatrix& edge_ext_agg,
                                         const SparseMatrix<T>& agg_edge_ext,
                                         const SparseMatrix<double>& face_int_agg_edge)
{
    const auto& agg_agg_offd = agg_agg.GetOffd();
    const auto& edge_ext_agg_offd = edge_ext_agg.GetOffd();

    int num_aggs = agg_agg_offd.Rows();
    int num_edges = face_int_agg_edge.Cols();
    int num_faces_int = face_int_agg_edge.Rows();
    int num_faces = num_faces_int + agg_agg_offd.nnz();

    for (int i = 0; i < 3; ++i)
    {
        if (agg_agg.GetMyId() == i)
        {
            printf("PRocs: %d\n", i);
            agg_agg.GetDiag().ToDense().Print("agg agg diag", std::cout, 4, 0);
            agg_agg.GetOffd().ToDense().Print("agg agg offd", std::cout, 4, 0);
        }
        MPI_Barrier(agg_agg.GetComm());
    }
    printf("%d Num Faces: %d\n", agg_agg.GetMyId(), num_faces);

    std::vector<int> indptr;
    std::vector<int> indices;

    indptr.reserve(num_faces + 1);

    const auto& ext_indptr = face_int_agg_edge.GetIndptr();
    const auto& ext_indices = face_int_agg_edge.GetIndices();
    const auto& ext_data = face_int_agg_edge.GetData();

    indptr.push_back(0);

    for (int i = 0; i < num_faces_int; i++)
    {
        for (int j = ext_indptr[i]; j < ext_indptr[i + 1]; j++)
        {
            if (ext_data[j] > 1.5)
            {
                indices.push_back(ext_indices[j]);
            }
        }

        indptr.push_back(indices.size());
    }

    const auto& agg_edge_indptr = agg_edge_ext.GetIndptr();
    const auto& agg_edge_indices = agg_edge_ext.GetIndices();

    const auto& agg_offd_indptr = agg_agg_offd.GetIndptr();
    const auto& agg_offd_indices = agg_agg_offd.GetIndices();
    const auto& agg_colmap = agg_agg.GetColMap();

    const auto& edge_offd_indptr = edge_ext_agg_offd.GetIndptr();
    const auto& edge_offd_indices = edge_ext_agg_offd.GetIndices();
    const auto& edge_colmap = edge_ext_agg.GetColMap();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            int shared = agg_colmap[agg_offd_indices[j]];

            for (int k = agg_edge_indptr[i]; k < agg_edge_indptr[i + 1]; ++k)
            {
                int edge = agg_edge_indices[k];

                if (edge_ext_agg_offd.RowSize(edge) > 0)
                {
                    int edge_loc = edge_offd_indices[edge_offd_indptr[edge]];

                    if (edge_colmap[edge_loc] == shared)
                    {
                        indices.push_back(edge);
                    }
                }
            }

            indptr.push_back(indices.size());
        }
    }

    assert(static_cast<int>(indptr.size()) == num_faces + 1);

    std::vector<double> data(indices.size(), 1.0);

    return SparseMatrix<double>(std::move(indptr), std::move(indices), std::move(data),
                           num_faces, num_edges);
}

template <typename T>
SparseMatrix<T> GraphTopology<T>::ExtendFaceAgg(const ParMatrix& agg_agg,
                                                const SparseMatrix<double>& face_int_agg)
{
    const auto& agg_agg_offd = agg_agg.GetOffd();

    int num_aggs = agg_agg.Rows();

    std::vector<int> indptr(face_int_agg.GetIndptr());
    std::vector<int> indices(face_int_agg.GetIndices());

    const auto& agg_offd_indptr = agg_agg_offd.GetIndptr();

    for (int i = 0; i < num_aggs; ++i)
    {
        for (int j = agg_offd_indptr[i]; j < agg_offd_indptr[i + 1]; ++j)
        {
            indices.push_back(i);
            indptr.push_back(indices.size());
        }
    }

    int num_faces = indptr.size() - 1;

    std::vector<T> data(indices.size(), 1.0);

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                        num_faces, num_aggs);
}

} // namespace linalgcpp

#endif /* GRAPHTOPOLOGY_HPP */
