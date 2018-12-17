#include "graph_partition.hpp"

namespace linalgcpp
{

double ComputeT(const linalgcpp::SparseMatrix<double>& A)
{
    const auto& data = A.GetData();

    return std::accumulate(std::begin(data), std::end(data), 0.0);
}

ParMatrix ParPartition(const ParMatrix& A_fine, int num_parts)
{
    MPI_Comm comm = A_fine.GetComm();
    int myid = A_fine.GetMyId();

    double T = ComputeT(A_fine);

    int num_vertices = A_fine.Rows();

    std::vector<double> alpha(num_vertices, 0);
    {
        const auto& diag_indptr = A_fine.GetDiag().GetIndptr();
        const auto& diag_indices = A_fine.GetDiag().GetIndices();
        const auto& diag_data = A_fine.GetDiag().GetData();

        const auto& offd_indptr = A_fine.GetOffd().GetIndptr();
        const auto& offd_indices = A_fine.GetOffd().GetIndices();
        const auto& offd_data = A_fine.GetOffd().GetData();

        for (int i = 0; i < num_vertices; ++i)
        {
            double diag_total = std::accumulate(std::begin(diag_data) + diag_indptr[i],
                    std::begin(diag_data) + diag_indptr[i + 1], 0.0);

            double offd_total = std::accumulate(std::begin(offd_data) + offd_indptr[i],
                    std::begin(offd_data) + offd_indptr[i + 1], 0.0);

            alpha[i] = (diag_total + offd_total) / T;
        }
    }

    ParMatrix A_i(A_fine);
    ParMatrix PT(comm, A_fine.GetRowStarts(), SparseIdentity(num_vertices));

    int iter = 0;
    int global_vertices = A_i.GlobalRows();
    int global_vertices_prev = -1;

    while (A_i.GlobalRows() > num_parts && global_vertices != global_vertices_prev)
    {
        std::vector<int> local_indices(A_i.Rows());
        std::iota(std::begin(local_indices), std::end(local_indices), A_i.GetRowStarts()[0]);

        auto comm_pkg = A_i.MakeCommPkg();

        auto global_indices = Broadcast(comm_pkg, local_indices);
        auto alpha_offd = Broadcast(comm_pkg, alpha);

        //if (myid == 0)
        //{
        //    std::cout << std::setprecision(4) << "0 Alpha diag: " << alpha;
        //    std::cout << std::setprecision(4) << "0 Alpha offd: " << alpha_offd;
        //}
        //MPI_Barrier(comm);

        //if (myid == 1)
        //{
        //    std::cout << std::setprecision(4) << "1 Alpha diag: " << alpha;
        //    std::cout << std::setprecision(4) << "1 Alpha offd: " << alpha_offd;
        //}
        //MPI_Barrier(comm);

        int num_vertices = A_i.Rows();

        const auto& diag_indptr = A_i.GetDiag().GetIndptr();
        const auto& diag_indices = A_i.GetDiag().GetIndices();
        const auto& diag_data = A_i.GetDiag().GetData();

        const auto& offd_indptr = A_i.GetOffd().GetIndptr();
        const auto& offd_indices = A_i.GetOffd().GetIndices();
        const auto& offd_data = A_i.GetOffd().GetData();
        const auto& col_map = A_i.GetColMap();

        std::vector<int> indices(num_vertices, -1);

        std::vector<int> P_diag_indptr(1, 0);
        std::vector<int> P_diag_indices;

        std::vector<int> P_offd_indptr(1, 0);
        std::vector<int> P_offd_indices;

        std::vector<int> P_col_map;

        double is_selected = 1.0;
        double not_selected = -1.0;

        std::vector<int> selected_local(A_i.Rows(), not_selected);
        int num_iter = 2;

        for (int iter = 0; iter < num_iter; ++iter)
        {
            bool last_iter = (iter == num_iter - 1);

            auto selected_offd = Broadcast(comm_pkg, selected_local);

            for (int i = 0; i < num_vertices; ++i)
            {
                //double score = -std::numeric_limits<double>::infinity();
                double score = 0.0;
                int index = -1;

                double alpha_i = alpha[i];

                if (selected_local[i] == is_selected)
                {
                    continue;
                }

                for (int k = diag_indptr[i]; k < diag_indptr[i + 1]; ++k)
                {
                    int local_col = diag_indices[k];

                    if (selected_local[local_col] == not_selected && local_col !=i)
                    {
                        double a_ij = diag_data[k];
                        double alpha_j = alpha[local_col];
                        double q_ij = 2.0 * (a_ij / T - alpha_i * alpha_j);

                        if (q_ij > score)
                        {
                            score = q_ij;
                            index = A_i.GetColStarts()[0] + local_col;
                        }
                        else if (q_ij == score)
                        {
                            index = -1;
                        }
                    }
                }

                for (int k = offd_indptr[i]; k < offd_indptr[i + 1]; ++k)
                {
                    int local_col = offd_indices[k];
                    int global_col = col_map[local_col];

                    double a_ij = offd_data[k];
                    auto it = std::lower_bound(std::begin(global_indices), std::end(global_indices), global_col);
                    assert(*it == global_col);

                    int alpha_index = it - std::begin(global_indices);

                    if (selected_offd[alpha_index] == is_selected)
                    {
                        continue;
                    }

                    double alpha_j = alpha_offd[alpha_index];
                    double q_ij = 2.0 * (a_ij / T - alpha_i * alpha_j);

                    if (q_ij > score)
                    {
                        score = q_ij;
                        index = global_col;
                    }
                    else if (q_ij == score)
                    {
                        index = -1;
                    }
                }

                int global_i = i + A_i.GetColStarts()[0];
                //printf("%d - global_i: %d Score %d %.8f \n", myid, global_i, index, score);

                indices[i] = index;
            }

            auto indices_offd = Broadcast(comm_pkg, indices);

            for (int i = 0; i < num_vertices; ++i)
            {
                int local_i = i;
                int global_j = indices[i];
                int global_i = A_i.GetColStarts()[0] + i;

                if (global_j != -1)
                {
                    if (!(global_j < A_i.GetColStarts()[0] || global_j >= A_i.GetColStarts()[1]))
                    {
                        //inside diag
                        int local_j = global_j - A_i.GetColStarts()[0];
                        int global_k = indices[local_j];

                        if (global_i == global_k)
                        {
                            if (global_i < global_j)
                            {
                                //printf("%d : Merge Inside %d %d\n", myid, global_i, global_j);
                                // merge 2 inside
                                if (selected_local[local_i] == not_selected)
                                {
                                    assert(selected_local[local_j] == not_selected);

                                    P_diag_indices.push_back(local_i);
                                    P_diag_indices.push_back(local_j);

                                    P_diag_indptr.push_back(P_diag_indices.size());
                                    P_offd_indptr.push_back(P_offd_indices.size());

                                    selected_local[local_i] = is_selected;
                                    selected_local[local_j] = is_selected;
                                }
                            }
                        }
                        else
                        {
                            //printf("%d : Identity, Point Inside %d %d %d\n", myid, global_i, global_j, global_k);
                            if (last_iter && selected_local[local_i] == not_selected)
                            {
                                P_diag_indices.push_back(local_i);

                                P_diag_indptr.push_back(P_diag_indices.size());
                                P_offd_indptr.push_back(P_offd_indices.size());
                            }
                        }
                    }
                    else
                    {
                        //outside diag
                        auto it = std::lower_bound(std::begin(global_indices), std::end(global_indices), global_j);
                        int global_k_index = it - std::begin(global_indices);

                        int global_k = indices_offd[global_k_index];

                        if (global_i == global_k)
                        {
                            // Processors own merges greater than their offsets
                            if (global_i < global_j)
                            {
                                if (selected_local[local_i] == not_selected)
                                {
                                    auto it = std::lower_bound(std::begin(global_indices), std::end(global_indices), global_j);
                                    int global_j_index = it - std::begin(global_indices);
                                    assert(selected_offd[global_j_index] == not_selected);

                                    //printf("%d : Merge Offd %d %d\n", myid, global_i, global_j);
                                    int local_j = P_offd_indices.size();

                                    P_diag_indices.push_back(local_i);

                                    P_offd_indices.push_back(local_j);
                                    P_col_map.push_back(global_j);

                                    P_diag_indptr.push_back(P_diag_indices.size());
                                    P_offd_indptr.push_back(P_offd_indices.size());

                                    selected_local[local_i] = is_selected;
                                }
                            }
                            else
                            {
                                selected_local[local_i] = is_selected;
                                //printf("%d : Do Nothing Offd %d %d\n", myid, global_i, global_j);
                                // Do nothing (hopefully)
                            }
                        }
                        else
                        {
                            //printf("%d : Identity, Point offd %d \n", myid, global_i);
                            if (last_iter && selected_local[local_i] == not_selected)
                            {
                                P_diag_indices.push_back(local_i);

                                P_diag_indptr.push_back(P_diag_indices.size());
                                P_offd_indptr.push_back(P_offd_indices.size());
                            }
                        }
                    }
                }
                else
                {
                    //printf("%d : Index negative 1 %d \n", myid, global_i);
                    if (last_iter && selected_local[local_i] == not_selected)
                    {
                        P_diag_indices.push_back(local_i);

                        P_diag_indptr.push_back(P_diag_indices.size());
                        P_offd_indptr.push_back(P_offd_indices.size());
                    }
                }
            }
        }


        SortColumnMap(P_col_map, P_offd_indices);
        std::vector<double> P_diag_data(P_diag_indices.size(), 1.0);
        std::vector<double> P_offd_data(P_offd_indices.size(), 1.0);

        linalgcpp_parverify(comm, P_diag_indptr.size() == P_offd_indptr.size());
        int diag_rows = P_diag_indptr.size() - 1;

        linalgcpp::SparseMatrix<double> P_diag(std::move(P_diag_indptr), std::move(P_diag_indices),
                                               std::move(P_diag_data),
                                               diag_rows, num_vertices);

        linalgcpp::SparseMatrix<double> P_offd(std::move(P_offd_indptr), std::move(P_offd_indices),
                                               std::move(P_offd_data),
                                               diag_rows, P_col_map.size());

        auto starts = linalgcpp::GenerateOffsets(comm, P_diag.Rows());

        ParMatrix PT_i(comm, starts, A_i.GetColStarts(),
                       std::move(P_diag), std::move(P_offd), P_col_map);
        ParMatrix P_i = PT_i.Transpose();

        //ParPrint(myid, printf("PT Iter: %d\n", iter)); iter++;
        //PrintDense(PT_i);


        PT = PT_i.Mult(PT);
        A_i = PT_i.Mult(A_i).Mult(P_i);

        bool check_Q = false;
        double Q;

        if (check_Q)
        {
            Q = CalcQ(A_fine, PT);

        }
        //A_i = hypre_test(A_i, P_i);
        global_vertices_prev = global_vertices;
        global_vertices = A_i.GlobalRows();

        VectorView<double> alpha_view(alpha);

        Vector<double> alpha_i = PT_i.Mult(alpha_view);

        alpha = std::vector<double>(std::begin(alpha_i), std::end(alpha_i));

        //int local_size = A_i.Rows();
        //int global_min = 0;
        //int global_max = 0;
        //MPI_Allreduce(&local_size, &global_min, 1, MPI_INT, MPI_MIN, comm);
        //MPI_Allreduce(&local_size, &global_max, 1, MPI_INT, MPI_MAX, comm);

        bool verbose = false;
        if (A_i.GetMyId() == 0 && verbose)
        {
            //std::cout << "Global Vertices :" << A_i.GlobalRows() << "\n";
            //std::cout << "Global Max Procs: " << global_max << " ";
            //std::cout << "Global Min Procs: " << global_min << " ";
            std::cout << "PT_i: " << PT_i.GlobalRows() << " ";
            std::cout << "Coarsening Factor: " << PT_i.GlobalRows() / (double) PT_i.GlobalCols() << " ";

            if (check_Q)
            {
                std::cout << "Q_i: " << Q << " ";
            }

            std::cout << "\n";
        }
    } // while

    return PT;
}

double CalcQ(const linalgcpp::SparseMatrix<double>& A, const linalgcpp::SparseMatrix<double>& P)
{
    assert(A.Cols() == P.Rows());
    assert(CheckSymmetric(A));

    double T = 0.0;

    int N = A.Rows();
    int M = P.Cols();

    std::vector<double> d(M, 0.0);
    std::vector<double> out(M, 0.0);

    const auto& indptr = A.GetIndptr();
    const auto& indices = A.GetIndices();
    const auto& data = A.GetData();

    const auto& agg = P.GetIndices();

    for (int i = 0; i < N; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            int col = indices[j];
            double val = data[j];

            int A = agg[i];
            int B = agg[col];

            if (A == B)
            {
                d[A] += val;
            }
            else
            {
                out[A] += val;
            }

            T += val;
        }
    }

    double Q = 0.0;

    for (int i = 0; i < M; ++i)
    {
        double alpha_i = (d[i] + out[i]) / T;

        Q += d[i] / T - alpha_i * alpha_i;
    }

    return Q;
}

double CalcQ(const ParMatrix& A, const ParMatrix& PT)
{
    ParMatrix P = PT.Transpose();
    auto I_A = UnDistributer(A);
    auto I_A_T = I_A.Transpose();
    auto I_P = UnDistributer(PT.Mult(P));

    auto A_0 = I_A_T.Mult(A).Mult(I_A).GetDiag();
    auto P_0 = I_A_T.Mult(P).Mult(I_P).GetDiag();

    double Q;
    if (A.GetMyId() == 0)
    {
        Q = CalcQ(A_0, P_0);
    }

    MPI_Bcast(&Q, 1, MPI_DOUBLE, 0, A.GetComm());

    return Q;
}

void SortColumnMap(std::vector<int>& col_map, std::vector<int>& indices)
{
    using linalgcpp::operator<<;

    const int col_map_size = col_map.size();

    std::vector<int> permutation(col_map_size);
    std::iota(std::begin(permutation), std::end(permutation), 0);

    auto compare = [&](int i, int j)
    {
        return col_map[i] < col_map[j];
    };

    std::sort(std::begin(permutation), std::end(permutation), compare);

    std::vector<int> col_index(col_map_size);

    for (int i = 0; i < col_map_size; ++i)
    {
        col_index[i] = indices[permutation[i]];
    }
    std::swap(indices, col_index);

    for (int i = 0; i < col_map_size; ++i)
    {
        col_index[i] = col_map[permutation[i]];
    }

    std::swap(col_map, col_index);
}

ParMatrix UnDistributer(const ParMatrix& A)
{
    MPI_Comm comm = A.GetComm();
    int myid = A.GetMyId();

    int diag_cols = (myid == 0) ? A.GlobalCols() : 0;
    int diag_rows = A.Rows();

    auto starts = linalgcpp::GenerateOffsets(comm, diag_cols);

    linalgcpp::SparseMatrix<double> diag;
    linalgcpp::SparseMatrix<double> offd;
    std::vector<int> col_map;

    if (myid == 0)
    {
        diag = SparseIdentity(diag_rows, diag_cols);
        offd = linalgcpp::SparseMatrix<double>(diag_rows, 0);
    }
    else
    {
        int offd_start = A.GetRowStarts()[0];
        int offd_cols = A.Rows();

        diag = linalgcpp::SparseMatrix<double>(diag_rows, 0);
        offd = SparseIdentity(diag_rows);

        col_map.resize(offd_cols);
        std::iota(std::begin(col_map), std::end(col_map), offd_start);
    }

    return ParMatrix(comm, A.GetRowStarts(), starts, diag, offd, col_map);
}

double ComputeT(const ParMatrix& A)
{
    double diag_T = ComputeT(A.GetDiag());
    double offd_T = ComputeT(A.GetOffd());

    double local_total = diag_T + offd_T;
    double global_total = 0.0;

    MPI_Allreduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_SUM, A.GetComm());

    return global_total;
}

void ParNormalizeRows(ParMatrix& PT)
{
    auto& diag_indptr = PT.GetDiag().GetIndptr();
    auto& diag_data = PT.GetDiag().GetData();
    auto& offd_indptr = PT.GetOffd().GetIndptr();
    auto& offd_data = PT.GetOffd().GetData();

    int num_rows = PT.Rows();

    for (int i = 0; i < num_rows; ++i)
    {
        double val = 1.0 / std::sqrt(PT.RowSize(i));

        for (int j = diag_indptr[i]; j < diag_indptr[i + 1]; ++j)
        {
            diag_data[j] = val;
        }

        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            offd_data[j] = val;
        }
    }
}

} // namespace linalgcpp
