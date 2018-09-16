/* A simple example of using linalgcpp
 */

#include "mpi.h"
#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"

using namespace linalgcpp;
using namespace parlinalgcpp;

void PrintVector(MPI_Comm comm, const std::vector<linalgcpp::Vector<double>>& vects, const std::vector<double>& evals);

int main(int argc, char** argv)
{
    int myid;
    int num_procs;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    /*
    SparseMatrix<double> diag(std::vector<double>(1, 1.0));
    SparseMatrix<double> offd(std::vector<double>(1, 1.0));

    std::vector<HYPRE_Int> col_maps(1, !myid);

    ParMatrix testp(comm, diag, offd, col_maps);
    ParMatrix test2 = testp.Mult(testp);
    */

    ///*
    assert(num_procs == 2);

    SparseMatrix<double> diag = ReadBinaryMat("test_data/test_mat.bin");
    std::vector<int> part = ReadBinaryVect<int>("test_data/test_part.bin");

    // Call LOBPCG with different preconditioners
    {
        ParMatrix pmat = ParSplit(comm, diag, part);
        pmat.AddDiag(1.0); // Make matrix nonsingular

        int num_vects = 2;
        int cg_max_iter = 1;
        int boomer_max_iter = 10;

        BoomerAMG boomer(pmat, boomer_max_iter);
        ParaSails parasails(pmat);

        ParDiagScale diag_scale(pmat);

        int relax_times = 4;
        ParSmoother test_smooth_copy;
        ParCG test_pcg_copy;

        ParSmoother smoother_gs(pmat, SmoothType::L1_GS, relax_times);
        ParSmoother smoother_jacobi(pmat, SmoothType::L1_Jacobi, relax_times);
        ParSmoother smoother_kz(pmat, SmoothType::Kaczmarz, relax_times);
        ParSmoother smoother_jac0(pmat, SmoothType::Jacobi, relax_times);
        ParSmoother smoother_gs_seq(pmat, SmoothType::GS_seq, relax_times);
        ParSmoother smoother_gs_par(pmat, SmoothType::GS_par, relax_times);
        ParSmoother smoother_hybrid_f(pmat, SmoothType::Hybrid_forward, relax_times);
        ParSmoother smoother_hybrid_b(pmat, SmoothType::Hybrid_backward, relax_times);
        ParSmoother smoother_hybrid_gs(pmat, SmoothType::Hybrid_GS, relax_times);
        ParSmoother smoother_hybrid_SSOR(pmat, SmoothType::Hybrid_SSOR, relax_times);
        ParSmoother smoother_jacobi_mv(pmat, SmoothType::Jacobi_matvec, relax_times);
        ParSmoother smoother_direct(pmat, SmoothType::Direct, relax_times);
        ParSmoother smoother_direct_blas(pmat, SmoothType::Direct_blas, relax_times);
        test_smooth_copy = smoother_gs;

        ParCG cg(pmat, cg_max_iter);

        ParCG pcg_boomer(pmat, boomer, cg_max_iter);
        ParCG pcg_parasails(pmat, parasails, cg_max_iter);

        ParCG pcg_diag(pmat, diag_scale, cg_max_iter);
        ParCG pcg_gs(pmat, smoother_gs, cg_max_iter);
        ParCG pcg_jacobi(pmat, smoother_jacobi, cg_max_iter);
        ParCG pcg_kz(pmat, smoother_kz, cg_max_iter);
        ParCG pcg_jac0(pmat, smoother_jac0, cg_max_iter);
        ParCG pcg_gs_seq(pmat, smoother_gs_seq, cg_max_iter);
        ParCG pcg_gs_par(pmat, smoother_gs_par, cg_max_iter);
        ParCG pcg_hybrid_f(pmat, smoother_hybrid_f, cg_max_iter);
        ParCG pcg_hybrid_b(pmat, smoother_hybrid_b, cg_max_iter);
        ParCG pcg_hybrid_gs(pmat, smoother_hybrid_gs, cg_max_iter);
        ParCG pcg_hybrid_SSOR(pmat, smoother_hybrid_SSOR, cg_max_iter);
        ParCG pcg_jacobi_mv(pmat, smoother_jacobi_mv, cg_max_iter);
        ParCG pcg_direct(pmat, smoother_direct, cg_max_iter);
        ParCG pcg_direct_blas(pmat, smoother_direct_blas, cg_max_iter);

        test_pcg_copy = cg;

        // rofl
        std::vector<const ParSolver*> ops {
            &cg,
            &pcg_boomer,
            &pcg_parasails,
            &pcg_diag,
            &pcg_gs,
            &pcg_jacobi,
            &pcg_kz,
            &boomer,
            &diag_scale,
            &smoother_gs,
            &smoother_jacobi,
            &smoother_kz,
            &smoother_jac0,
            &smoother_gs_seq,
            &smoother_gs_par,
            &smoother_hybrid_f,
            &smoother_hybrid_b,
            &smoother_hybrid_gs,
            &smoother_hybrid_SSOR,
            &smoother_jacobi_mv,
            &smoother_direct,
            //&smoother_direct_blas,  doesn't work right
            &pcg_gs_seq,
            &pcg_gs_par,
            &pcg_hybrid_f,
            &pcg_hybrid_b,
            &pcg_hybrid_gs,
            &pcg_hybrid_SSOR,
            &pcg_jacobi_mv,
            &pcg_direct,
            &pcg_jac0,
            &pcg_direct_blas,
            &test_smooth_copy,
            &test_pcg_copy,
        };

        //ops = {&smoother_direct_blas};

        std::vector<linalgcpp::Vector<double>> evects(num_vects, linalgcpp::Vector<double>(pmat.Rows()));

        int count = 0;
        for (const auto& op : ops)
        {
            if (myid == 0)
                printf("Op: %d\n", count++);

            for (auto& x_i : evects)
            {
                Randomize(x_i, -1.0, 1.0);
            }

            bool verbose = false;
            auto evals = LOBPCG(pmat, evects, op, verbose);

            PrintVector(comm, evects, evals);

            assert(std::fabs(evals[0] - 1.000000000000000) < 1e-10);
            assert(std::fabs(evals[1] - 1.295744016607458) < 1e-10);
        }
    }
//*/

    MPI_Finalize();

    return EXIT_SUCCESS;
}

void PrintVector(MPI_Comm comm, const std::vector<linalgcpp::Vector<double>>& evects, const std::vector<double>& evals)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    if (myid == 0)
    {
        std::cout.precision(10);
        std::cout << "Evals: " << evals;

        int subset_size = std::min(10, evects[0].size());
        for (int j = 0; j < subset_size; ++j)
        {
            for(size_t i = 0; i < evects.size(); ++i)
            {
                std::cout << evects[i][j] << "\t";
            }

            std::cout << "\n";
        }
    }

}
