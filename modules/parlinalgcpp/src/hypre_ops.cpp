/*! @file */
/*  BHEADER**********************************************************************
    Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
    Produced at the Lawrence Livermore National Laboratory.
    This file is part of HYPRE.  See file COPYRIGHT for details.

    HYPRE is free software; you can redistribute it and/or modify it under the
    terms of the GNU Lesser General Public License (as published by the Free
    Software Foundation) version 2.1 dated February 1999.

    $Revision$
 ***********************************************************************EHEADER*/

/*  BHEADER**********************************************************************
    Modifes Hypre routines so that memory allocated for matrices is in std::vector.
    This allows the serial blocks {diag, offd} to be created in place instead of copied.
    
    Modified from the original HYPRE code by gelever, November 20, 2017
    hypre-2.10.0b file: src/parcsr_mv/par_csr_matop.c
 ***********************************************************************EHEADER*/

#include "hypre_ops.hpp"

namespace parlinalgcpp
{

template <typename T = double>
void qsort0(T* data, int start, int end)
{
    std::sort(data + start, data + end + 1);
}

void RowSizes(
    std::vector<HYPRE_Int>& C_diag_i,
    std::vector<HYPRE_Int>& C_offd_i,
    const HYPRE_Int* A_diag_i,
    const HYPRE_Int* A_diag_j,
    const HYPRE_Int* A_offd_i,
    const HYPRE_Int* A_offd_j,
    const HYPRE_Int* B_diag_i,
    const HYPRE_Int* B_diag_j,
    const HYPRE_Int* B_offd_i,
    const HYPRE_Int* B_offd_j,
    const HYPRE_Int* B_ext_diag_i,
    const HYPRE_Int* B_ext_diag_j,
    const HYPRE_Int* B_ext_offd_i,
    const HYPRE_Int* B_ext_offd_j,
    const HYPRE_Int* map_B_to_C,
    HYPRE_Int* C_diag_size,
    HYPRE_Int* C_offd_size,
    HYPRE_Int num_rows_diag_A,
    HYPRE_Int num_cols_offd_A,
    HYPRE_Int allsquare,
    HYPRE_Int num_cols_diag_B,
    HYPRE_Int num_cols_offd_B,
    HYPRE_Int num_cols_offd_C
)
{

    HYPRE_Int i1, i2, i3, jj2, jj3;
    HYPRE_Int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
    HYPRE_Int start_indexing = 0; /* start indexing for C_data at 0 */
    HYPRE_Int num_threads = hypre_NumThreads();
    HYPRE_Int* jj_count_diag_array;
    HYPRE_Int* jj_count_offd_array;
    HYPRE_Int ii, size, rest;
    /*  First pass begins here.  Computes sizes of C rows.
        Arrays computed: C_diag_i, C_offd_i, B_marker
        Arrays needed: (11, all HYPRE_Int*)
        A_diag_i, A_diag_j, A_offd_i, A_offd_j,
        B_diag_i, B_diag_j, B_offd_i, B_offd_j,
        B_ext_i, B_ext_j, col_map_offd_B,
        col_map_offd_B, B_offd_i, B_offd_j, B_ext_i, B_ext_j,
        Scalars computed: C_diag_size, C_offd_size
        Scalars needed:
        num_rows_diag_A, num_rows_diag_A, num_cols_offd_A, allsquare,
        first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
    */

    C_diag_i.resize(num_rows_diag_A + 1);
    C_offd_i.resize(num_rows_diag_A + 1);

    jj_count_diag_array = hypre_CTAlloc(HYPRE_Int, num_threads);
    jj_count_offd_array = hypre_CTAlloc(HYPRE_Int, num_threads);
    /*  -----------------------------------------------------------------------
        Loop over rows of A
        -----------------------------------------------------------------------*/
    size = num_rows_diag_A / num_threads;
    rest = num_rows_diag_A - size * num_threads;
#ifdef HYPRE_USING_OPENMP
    #pragma omp parallel private(ii, i1, jj_row_begin_diag, jj_row_begin_offd, jj_count_diag, jj_count_offd, jj2, i2, jj3, i3)
#endif
    /*for (ii=0; ii < num_threads; ii++)*/
    {
        HYPRE_Int* B_marker = NULL;
        HYPRE_Int ns, ne;
        ii = hypre_GetThreadNum();

        if (ii < rest)
        {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
        }
        else
        {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
        }

        jj_count_diag = start_indexing;
        jj_count_offd = start_indexing;

        if (num_cols_diag_B || num_cols_offd_C)
        {
            B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B + num_cols_offd_C);
        }

        for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
        {
            B_marker[i1] = -1;
        }

        for (i1 = ns; i1 < ne; i1++)
        {
            /*  --------------------------------------------------------------------
                Set marker for diagonal entry, C_{i1,i1} (for square matrices).
                --------------------------------------------------------------------*/

            jj_row_begin_diag = jj_count_diag;
            jj_row_begin_offd = jj_count_offd;

            if ( allsquare )
            {
                B_marker[i1] = jj_count_diag;
                jj_count_diag++;
            }

            /*  -----------------------------------------------------------------
                Loop over entries in row i1 of A_offd.
                -----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
                for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
                {
                    i2 = A_offd_j[jj2];

                    /*  -----------------------------------------------------------
                        Loop over entries in row i2 of B_ext.
                        -----------------------------------------------------------*/

                    for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
                    {
                        i3 = num_cols_diag_B + B_ext_offd_j[jj3];

                        /*  --------------------------------------------------------
                            Check B_marker to see that C_{i1,i3} has not already
                            been accounted for. If it has not, mark it and increment
                            counter.
                            --------------------------------------------------------*/

                        if (B_marker[i3] < jj_row_begin_offd)
                        {
                            B_marker[i3] = jj_count_offd;
                            jj_count_offd++;
                        }
                    }

                    for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
                    {
                        i3 = B_ext_diag_j[jj3];

                        if (B_marker[i3] < jj_row_begin_diag)
                        {
                            B_marker[i3] = jj_count_diag;
                            jj_count_diag++;
                        }
                    }
                }
            }

            /*  -----------------------------------------------------------------
                Loop over entries in row i1 of A_diag.
                -----------------------------------------------------------------*/

            for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
            {
                i2 = A_diag_j[jj2];

                /*  -----------------------------------------------------------
                    Loop over entries in row i2 of B_diag.
                    -----------------------------------------------------------*/

                for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
                {
                    i3 = B_diag_j[jj3];

                    /*  --------------------------------------------------------
                        Check B_marker to see that C_{i1,i3} has not already
                        been accounted for. If it has not, mark it and increment
                        counter.
                        --------------------------------------------------------*/

                    if (B_marker[i3] < jj_row_begin_diag)
                    {
                        B_marker[i3] = jj_count_diag;
                        jj_count_diag++;
                    }
                }

                /*  -----------------------------------------------------------
                    Loop over entries in row i2 of B_offd.
                    -----------------------------------------------------------*/

                if (num_cols_offd_B)
                {
                    for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
                    {
                        i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

                        /*  --------------------------------------------------------
                            Check B_marker to see that C_{i1,i3} has not already
                            been accounted for. If it has not, mark it and increment
                            counter.
                            --------------------------------------------------------*/

                        if (B_marker[i3] < jj_row_begin_offd)
                        {
                            B_marker[i3] = jj_count_offd;
                            jj_count_offd++;
                        }
                    }
                }
            }

            /*  --------------------------------------------------------------------
                Set C_diag_i and C_offd_i for this row.
                --------------------------------------------------------------------*/

            C_diag_i[i1] = jj_row_begin_diag;
            C_offd_i[i1] = jj_row_begin_offd;

        }

        jj_count_diag_array[ii] = jj_count_diag;
        jj_count_offd_array[ii] = jj_count_offd;

        hypre_TFree(B_marker);
#ifdef HYPRE_USING_OPENMP
        #pragma omp barrier
#endif

        if (ii)
        {
            jj_count_diag = jj_count_diag_array[0];
            jj_count_offd = jj_count_offd_array[0];

            for (i1 = 1; i1 < ii; i1++)
            {
                jj_count_diag += jj_count_diag_array[i1];
                jj_count_offd += jj_count_offd_array[i1];
            }

            for (i1 = ns; i1 < ne; i1++)
            {
                C_diag_i[i1] += jj_count_diag;
                C_offd_i[i1] += jj_count_offd;
            }
        }
        else
        {
            C_diag_i[num_rows_diag_A] = 0;
            C_offd_i[num_rows_diag_A] = 0;

            for (i1 = 0; i1 < num_threads; i1++)
            {
                C_diag_i[num_rows_diag_A] += jj_count_diag_array[i1];
                C_offd_i[num_rows_diag_A] += jj_count_offd_array[i1];
            }
        }
    } /* end parallel loop */

    /*  -----------------------------------------------------------------------
        Allocate C_diag_data and C_diag_j arrays.
        Allocate C_offd_data and C_offd_j arrays.
        -----------------------------------------------------------------------*/

    *C_diag_size = C_diag_i[num_rows_diag_A];
    *C_offd_size = C_offd_i[num_rows_diag_A];

    hypre_TFree(jj_count_diag_array);
    hypre_TFree(jj_count_offd_array);

    /* End of First Pass */
}

ParMatrix ParMatrix::Mult(const ParMatrix& other) const
{
    hypre_ParCSRMatrix* A = A_;
    hypre_ParCSRMatrix* B = other.A_;

    MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

    hypre_CSRMatrix* A_diag = hypre_ParCSRMatrixDiag(A);

    HYPRE_Complex*   A_diag_data = hypre_CSRMatrixData(A_diag);
    HYPRE_Int*       A_diag_i = hypre_CSRMatrixI(A_diag);
    HYPRE_Int*       A_diag_j = hypre_CSRMatrixJ(A_diag);

    hypre_CSRMatrix* A_offd = hypre_ParCSRMatrixOffd(A);

    HYPRE_Complex*   A_offd_data = hypre_CSRMatrixData(A_offd);
    HYPRE_Int*       A_offd_i = hypre_CSRMatrixI(A_offd);
    HYPRE_Int*       A_offd_j = hypre_CSRMatrixJ(A_offd);

    HYPRE_Int        num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
    HYPRE_Int        num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
    HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

    hypre_CSRMatrix* B_diag = hypre_ParCSRMatrixDiag(B);

    HYPRE_Complex*   B_diag_data = hypre_CSRMatrixData(B_diag);
    HYPRE_Int*       B_diag_i = hypre_CSRMatrixI(B_diag);
    HYPRE_Int*       B_diag_j = hypre_CSRMatrixJ(B_diag);

    hypre_CSRMatrix* B_offd = hypre_ParCSRMatrixOffd(B);
    HYPRE_Int*       col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

    HYPRE_Complex*   B_offd_data = hypre_CSRMatrixData(B_offd);
    HYPRE_Int*       B_offd_i = hypre_CSRMatrixI(B_offd);
    HYPRE_Int*       B_offd_j = hypre_CSRMatrixJ(B_offd);

    HYPRE_Int        first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
    HYPRE_Int        last_col_diag_B;
    HYPRE_Int        num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
    HYPRE_Int        num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
    HYPRE_Int        num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

    std::vector<HYPRE_Int> col_map_offd_C;
    HYPRE_Int*          map_B_to_C = NULL;

    std::vector<double> C_diag_data;
    std::vector<int> C_diag_i;
    std::vector<int> C_diag_j;

    std::vector<double> C_offd_data;
    std::vector<int> C_offd_i;
    std::vector<int> C_offd_j;

    HYPRE_Int        C_diag_size;
    HYPRE_Int        C_offd_size;
    HYPRE_Int        num_cols_offd_C = 0;

    hypre_CSRMatrix* Bs_ext = nullptr;

    HYPRE_Complex*   Bs_ext_data = nullptr;
    HYPRE_Int*       Bs_ext_i = nullptr;
    HYPRE_Int*       Bs_ext_j = nullptr;

    HYPRE_Complex*   B_ext_diag_data = nullptr;
    HYPRE_Int*       B_ext_diag_i = nullptr;
    HYPRE_Int*       B_ext_diag_j = nullptr;
    HYPRE_Int        B_ext_diag_size;

    HYPRE_Complex*   B_ext_offd_data = nullptr;
    HYPRE_Int*       B_ext_offd_i = nullptr;
    HYPRE_Int*       B_ext_offd_j = nullptr;
    HYPRE_Int        B_ext_offd_size;

    HYPRE_Int*       temp = nullptr;

    HYPRE_Int        n_cols_A;
    HYPRE_Int        n_rows_B;
    HYPRE_Int        allsquare = 0;
    HYPRE_Int        cnt;
    HYPRE_Int        num_procs;
    HYPRE_Int        value;
    HYPRE_Int*       my_diag_array;
    HYPRE_Int*       my_offd_array;
    HYPRE_Int        max_num_threads;

    HYPRE_Complex    zero = 0.0;

    n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
    n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);

    max_num_threads = hypre_NumThreads();
    my_diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads);
    my_offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads);

    assert(n_cols_A == n_rows_B && num_cols_diag_A == num_rows_diag_B);

    if ( num_rows_diag_A == num_cols_diag_B)
    {
        allsquare = 1;
    }

    /*  -----------------------------------------------------------------------
        Extract B_ext, i.e. portion of B that is stored on neighbor procs
        and needed locally for matrix matrix product
        -----------------------------------------------------------------------*/

    hypre_MPI_Comm_size(comm, &num_procs);

    if (num_procs > 1)
    {
        /*  ---------------------------------------------------------------------
            If there exists no CommPkg for A, a CommPkg is generated using
            equally load balanced partitionings within
            hypre_ParCSRMatrixExtractBExt
            --------------------------------------------------------------------*/
        Bs_ext = hypre_ParCSRMatrixExtractBExt(B, A, 1);
        Bs_ext_data = hypre_CSRMatrixData(Bs_ext);
        Bs_ext_i    = hypre_CSRMatrixI(Bs_ext);
        Bs_ext_j    = hypre_CSRMatrixJ(Bs_ext);
    }

    B_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1);
    B_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1);
    B_ext_diag_size = 0;
    B_ext_offd_size = 0;
    last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

#ifdef HYPRE_USING_OPENMP
    #pragma omp parallel
#endif
    {
        HYPRE_Int size, rest, ii;
        HYPRE_Int ns, ne;
        HYPRE_Int i1, i, j;
        HYPRE_Int my_offd_size, my_diag_size;
        HYPRE_Int cnt_offd, cnt_diag;
        HYPRE_Int num_threads = hypre_NumActiveThreads();

        size = num_cols_offd_A / num_threads;
        rest = num_cols_offd_A - size * num_threads;
        ii = hypre_GetThreadNum();

        if (ii < rest)
        {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
        }
        else
        {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
        }

        my_diag_size = 0;
        my_offd_size = 0;

        for (i = ns; i < ne; i++)
        {
            B_ext_diag_i[i] = my_diag_size;
            B_ext_offd_i[i] = my_offd_size;

            for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
                if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
                {
                    my_offd_size++;
                }
                else
                {
                    my_diag_size++;
                }
        }

        my_diag_array[ii] = my_diag_size;
        my_offd_array[ii] = my_offd_size;

#ifdef HYPRE_USING_OPENMP
        #pragma omp barrier
#endif

        if (ii)
        {
            my_diag_size = my_diag_array[0];
            my_offd_size = my_offd_array[0];

            for (i1 = 1; i1 < ii; i1++)
            {
                my_diag_size += my_diag_array[i1];
                my_offd_size += my_offd_array[i1];
            }

            for (i1 = ns; i1 < ne; i1++)
            {
                B_ext_diag_i[i1] += my_diag_size;
                B_ext_offd_i[i1] += my_offd_size;
            }
        }
        else
        {
            B_ext_diag_size = 0;
            B_ext_offd_size = 0;

            for (i1 = 0; i1 < num_threads; i1++)
            {
                B_ext_diag_size += my_diag_array[i1];
                B_ext_offd_size += my_offd_array[i1];
            }

            B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
            B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

            if (B_ext_diag_size)
            {
                B_ext_diag_j = hypre_CTAlloc(HYPRE_Int, B_ext_diag_size);
                B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size);
            }

            if (B_ext_offd_size)
            {
                B_ext_offd_j = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size);
                B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size);
            }

            if (B_ext_offd_size || num_cols_offd_B)
            {
                temp = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size + num_cols_offd_B);
            }
        }

#ifdef HYPRE_USING_OPENMP
        #pragma omp barrier
#endif

        cnt_offd = B_ext_offd_i[ns];
        cnt_diag = B_ext_diag_i[ns];

        for (i = ns; i < ne; i++)
        {
            for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
                if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
                {
                    temp[cnt_offd] = Bs_ext_j[j];
                    B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
                    B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
                }
                else
                {
                    B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
                    B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
                }
        }

#ifdef HYPRE_USING_OPENMP
        #pragma omp barrier
#endif

        if (ii == 0)
        {

            if (num_procs > 1)
            {
                hypre_CSRMatrixDestroy(Bs_ext);
                Bs_ext = NULL;
            }

            cnt = 0;

            if (B_ext_offd_size || num_cols_offd_B)
            {
                cnt = B_ext_offd_size;

                for (i = 0; i < num_cols_offd_B; i++)
                {
                    temp[cnt++] = col_map_offd_B[i];
                }

                if (cnt)
                {
                    qsort0(temp, 0, cnt - 1);
                    num_cols_offd_C = 1;
                    value = temp[0];

                    for (i = 1; i < cnt; i++)
                    {
                        if (temp[i] > value)
                        {
                            value = temp[i];
                            temp[num_cols_offd_C++] = value;
                        }
                    }
                }

                col_map_offd_C.resize(num_cols_offd_C);

                for (i = 0; i < num_cols_offd_C; i++)
                {
                    col_map_offd_C[i] = temp[i];
                }

                hypre_TFree(temp);
            }
        }

#ifdef HYPRE_USING_OPENMP
        #pragma omp barrier
#endif

        for (i = ns; i < ne; i++)
            for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i + 1]; j++)
                B_ext_offd_j[j] = hypre_BinarySearch(col_map_offd_C.data(), B_ext_offd_j[j],
                                                     num_cols_offd_C);

    } /* end parallel region */

    hypre_TFree(my_diag_array);
    hypre_TFree(my_offd_array);

    if (num_cols_offd_B)
    {
        HYPRE_Int i;
        map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B);

        cnt = 0;

        for (i = 0; i < num_cols_offd_C; i++)
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
                map_B_to_C[cnt++] = i;

                if (cnt == num_cols_offd_B)
                {
                    break;
                }
            }
    }

    RowSizes(
        C_diag_i, C_offd_i,
        A_diag_i, A_diag_j, A_offd_i, A_offd_j,
        B_diag_i, B_diag_j, B_offd_i, B_offd_j,
        B_ext_diag_i, B_ext_diag_j, B_ext_offd_i, B_ext_offd_j,
        map_B_to_C,
        &C_diag_size, &C_offd_size,
        num_rows_diag_A, num_cols_offd_A, allsquare,
        num_cols_diag_B, num_cols_offd_B,
        num_cols_offd_C
    );

    /*  -----------------------------------------------------------------------
        Allocate C_diag_data and C_diag_j arrays.
        Allocate C_offd_data and C_offd_j arrays.
        -----------------------------------------------------------------------*/

    last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;
    C_diag_data.resize(C_diag_size);
    C_diag_j.resize(C_diag_size);

    if (C_offd_size)
    {
        C_offd_data.resize(C_offd_size);
        C_offd_j.resize(C_offd_size);
    }

    /*  -----------------------------------------------------------------------
        Second Pass: Fill in C_diag_data and C_diag_j.
        Second Pass: Fill in C_offd_data and C_offd_j.
        -----------------------------------------------------------------------*/

    /*  -----------------------------------------------------------------------
        Initialize some stuff.
        -----------------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
    #pragma omp parallel
#endif
    {
        HYPRE_Int* B_marker = NULL;
        HYPRE_Int ns, ne, size, rest, ii;
        HYPRE_Int i1, i2, i3, jj2, jj3;
        HYPRE_Int jj_row_begin_diag, jj_count_diag;
        HYPRE_Int jj_row_begin_offd, jj_count_offd;
        HYPRE_Int num_threads;
        HYPRE_Complex a_entry, a_b_product;

        ii = hypre_GetThreadNum();
        num_threads = hypre_NumActiveThreads();
        size = num_rows_diag_A / num_threads;
        rest = num_rows_diag_A - size * num_threads;

        if (ii < rest)
        {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
        }
        else
        {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
        }

        jj_count_diag = C_diag_i[ns];
        jj_count_offd = C_offd_i[ns];

        if (num_cols_diag_B || num_cols_offd_C)
        {
            B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B + num_cols_offd_C);
        }

        for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
        {
            B_marker[i1] = -1;
        }

        /*  -----------------------------------------------------------------------
            Loop over interior c-points.
            -----------------------------------------------------------------------*/

        for (i1 = ns; i1 < ne; i1++)
        {

            /*  --------------------------------------------------------------------
                Create diagonal entry, C_{i1,i1}
                --------------------------------------------------------------------*/

            jj_row_begin_diag = jj_count_diag;
            jj_row_begin_offd = jj_count_offd;

            if ( allsquare )
            {
                B_marker[i1] = jj_count_diag;
                C_diag_data[jj_count_diag] = zero;
                C_diag_j[jj_count_diag] = i1;
                jj_count_diag++;
            }

            /*  -----------------------------------------------------------------
                Loop over entries in row i1 of A_offd.
                -----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
                for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
                {
                    i2 = A_offd_j[jj2];
                    a_entry = A_offd_data[jj2];

                    /*  -----------------------------------------------------------
                        Loop over entries in row i2 of B_ext.
                        -----------------------------------------------------------*/

                    for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
                    {
                        i3 = num_cols_diag_B + B_ext_offd_j[jj3];
                        a_b_product = a_entry * B_ext_offd_data[jj3];

                        /*  --------------------------------------------------------
                            Check B_marker to see that C_{i1,i3} has not already
                            been accounted for. If it has not, create a new entry.
                            If it has, add new contribution.
                            --------------------------------------------------------*/

                        if (B_marker[i3] < jj_row_begin_offd)
                        {
                            B_marker[i3] = jj_count_offd;
                            C_offd_data[jj_count_offd] = a_b_product;
                            C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                            jj_count_offd++;
                        }
                        else
                        {
                            C_offd_data[B_marker[i3]] += a_b_product;
                        }
                    }

                    for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
                    {
                        i3 = B_ext_diag_j[jj3];
                        a_b_product = a_entry * B_ext_diag_data[jj3];

                        if (B_marker[i3] < jj_row_begin_diag)
                        {
                            B_marker[i3] = jj_count_diag;
                            C_diag_data[jj_count_diag] = a_b_product;
                            C_diag_j[jj_count_diag] = i3;
                            jj_count_diag++;
                        }
                        else
                        {
                            C_diag_data[B_marker[i3]] += a_b_product;
                        }
                    }
                }
            }

            /*  -----------------------------------------------------------------
                Loop over entries in row i1 of A_diag.
                -----------------------------------------------------------------*/

            for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
            {
                i2 = A_diag_j[jj2];
                a_entry = A_diag_data[jj2];

                /*  -----------------------------------------------------------
                    Loop over entries in row i2 of B_diag.
                    -----------------------------------------------------------*/

                for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
                {
                    i3 = B_diag_j[jj3];
                    a_b_product = a_entry * B_diag_data[jj3];

                    /*  --------------------------------------------------------
                        Check B_marker to see that C_{i1,i3} has not already
                        been accounted for. If it has not, create a new entry.
                        If it has, add new contribution.
                        --------------------------------------------------------*/

                    if (B_marker[i3] < jj_row_begin_diag)
                    {
                        B_marker[i3] = jj_count_diag;
                        C_diag_data[jj_count_diag] = a_b_product;
                        C_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
                    }
                    else
                    {
                        C_diag_data[B_marker[i3]] += a_b_product;
                    }
                }

                if (num_cols_offd_B)
                {
                    for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
                    {
                        i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];
                        a_b_product = a_entry * B_offd_data[jj3];

                        /*  --------------------------------------------------------
                            Check B_marker to see that C_{i1,i3} has not already
                            been accounted for. If it has not, create a new entry.
                            If it has, add new contribution.
                            --------------------------------------------------------*/

                        if (B_marker[i3] < jj_row_begin_offd)
                        {
                            B_marker[i3] = jj_count_offd;
                            C_offd_data[jj_count_offd] = a_b_product;
                            C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                            jj_count_offd++;
                        }
                        else
                        {
                            C_offd_data[B_marker[i3]] += a_b_product;
                        }
                    }
                }
            }
        }

        hypre_TFree(B_marker);
    } /*end parallel region */

    /*  -----------------------------------------------------------------------
        Free various arrays
        -----------------------------------------------------------------------*/

    hypre_TFree(B_ext_diag_i);

    if (B_ext_diag_size)
    {
        hypre_TFree(B_ext_diag_j);
        hypre_TFree(B_ext_diag_data);
    }

    hypre_TFree(B_ext_offd_i);

    if (B_ext_offd_size)
    {
        hypre_TFree(B_ext_offd_j);
        hypre_TFree(B_ext_offd_data);
    }

    if (num_cols_offd_B)
    {
        hypre_TFree(map_B_to_C);
    }

    const size_t num_cols_diag_C = other.col_starts_[1] - other.col_starts_[0];
    linalgcpp::SparseMatrix<double> C_diag(std::move(C_diag_i), std::move(C_diag_j), std::move(C_diag_data),
                                           num_rows_diag_A, num_cols_diag_C);
    linalgcpp::SparseMatrix<double> C_offd(std::move(C_offd_i), std::move(C_offd_j), std::move(C_offd_data),
                                           num_rows_diag_A, col_map_offd_C.size());

    return ParMatrix(comm, row_starts_, other.col_starts_,
                     std::move(C_diag), std::move(C_offd), std::move(col_map_offd_C));
}

ParMatrix ParMatrix::Transpose() const
{
    hypre_ParCSRMatrix* A = A_;
    HYPRE_Int data = 1;
    hypre_ParCSRCommHandle* comm_handle = nullptr;
    MPI_Comm comm = hypre_ParCSRMatrixComm(A);
    hypre_ParCSRCommPkg*  comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    hypre_CSRMatrix*      A_offd   = hypre_ParCSRMatrixOffd(A);
    HYPRE_Int  num_cols = hypre_ParCSRMatrixNumCols(A);
    HYPRE_Int  first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
    HYPRE_Int* row_starts = hypre_ParCSRMatrixRowStarts(A);
    HYPRE_Int* col_starts = hypre_ParCSRMatrixColStarts(A);

    HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
    HYPRE_Int        num_sends = 0, num_recvs = 0, num_cols_offd_AT;
    HYPRE_Int        i, j, k, index, counter, j_row;
    HYPRE_Int        value;

    HYPRE_Int* AT_tmp_i = nullptr;
    HYPRE_Int* AT_tmp_j = nullptr;
    HYPRE_Complex* AT_tmp_data = nullptr;

    HYPRE_Int* AT_buf_i = nullptr;
    HYPRE_Int* AT_buf_j = nullptr;
    HYPRE_Complex* AT_buf_data = nullptr;

    std::vector<HYPRE_Int> AT_offd_i;
    std::vector<HYPRE_Int> AT_offd_j;
    std::vector<HYPRE_Complex> AT_offd_data;

    std::vector<HYPRE_Int> col_map_offd_AT;
    std::vector<HYPRE_Int> row_starts_AT;
    std::vector<HYPRE_Int> col_starts_AT;

    HYPRE_Int num_procs, my_id;

    HYPRE_Int* recv_procs = nullptr;
    HYPRE_Int* send_procs = nullptr;
    HYPRE_Int* recv_vec_starts = nullptr;
    HYPRE_Int* send_map_starts = nullptr;
    HYPRE_Int* send_map_elmts = nullptr;
    HYPRE_Int* tmp_recv_vec_starts = nullptr;
    HYPRE_Int* tmp_send_map_starts = nullptr;
    hypre_ParCSRCommPkg* tmp_comm_pkg = nullptr;

    hypre_MPI_Comm_size(comm, &num_procs);
    hypre_MPI_Comm_rank(comm, &my_id);

    num_cols_offd_AT = 0;
    counter = 0;

    /*  ---------------------------------------------------------------------
        If there exists no CommPkg for A, a CommPkg is generated using
        equally load balanced partitionings
        --------------------------------------------------------------------*/

    if (!comm_pkg)
    {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    }

    AT_offd_i.resize(num_cols + 1);

    if (num_procs > 1)
    {
        linalgcpp::SparseMatrix<double> A_offd_T = offd_.Transpose();

        AT_tmp_i = A_offd_T.GetIndptr().data();
        AT_tmp_j = A_offd_T.GetIndices().data();
        AT_tmp_data = A_offd_T.GetData().data();

        num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
        num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
        recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
        send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
        recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
        send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
        send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

        AT_buf_i = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends]);

        for (i = 0; i < AT_tmp_i[num_cols_offd]; i++)
        {
            AT_tmp_j[i] += first_row_index;
        }

        for (i = 0; i < num_cols_offd; i++)
        {
            AT_tmp_i[i] = AT_tmp_i[i + 1] - AT_tmp_i[i];
        }

        comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, AT_tmp_i, AT_buf_i);
        hypre_ParCSRCommHandleDestroy(comm_handle);
        comm_handle = NULL;

        tmp_send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends + 1);
        tmp_recv_vec_starts = hypre_CTAlloc(HYPRE_Int, num_recvs + 1);

        tmp_send_map_starts[0] = send_map_starts[0];

        for (i = 0; i < num_sends; i++)
        {
            tmp_send_map_starts[i + 1] = tmp_send_map_starts[i];

            for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
            {
                tmp_send_map_starts[i + 1] += AT_buf_i[j];
                AT_offd_i[send_map_elmts[j] + 1] += AT_buf_i[j];
            }
        }

        for (i = 0; i < num_cols; i++)
        {
            AT_offd_i[i + 1] += AT_offd_i[i];
        }

        tmp_recv_vec_starts[0] = recv_vec_starts[0];

        for (i = 0; i < num_recvs; i++)
        {
            tmp_recv_vec_starts[i + 1] = tmp_recv_vec_starts[i];

            for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
            {
                tmp_recv_vec_starts[i + 1] +=  AT_tmp_i[j];
            }
        }

        tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);
        hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
        hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
        hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
        hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = recv_procs;
        hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = send_procs;
        hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = tmp_recv_vec_starts;
        hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = tmp_send_map_starts;

        AT_buf_j = hypre_CTAlloc(HYPRE_Int, tmp_send_map_starts[num_sends]);
        comm_handle = hypre_ParCSRCommHandleCreate(12, tmp_comm_pkg, AT_tmp_j,
                                                   AT_buf_j);
        hypre_ParCSRCommHandleDestroy(comm_handle);
        comm_handle = NULL;

        if (data)
        {
            AT_buf_data = hypre_CTAlloc(HYPRE_Complex, tmp_send_map_starts[num_sends]);
            comm_handle = hypre_ParCSRCommHandleCreate(2, tmp_comm_pkg, AT_tmp_data,
                                                       AT_buf_data);
            hypre_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
        }

        hypre_TFree(tmp_recv_vec_starts);
        hypre_TFree(tmp_send_map_starts);
        hypre_TFree(tmp_comm_pkg);

        if (AT_offd_i[num_cols])
        {
            AT_offd_j.resize(AT_offd_i[num_cols]);
            AT_offd_data.resize(AT_offd_i[num_cols]);
        }

        counter = 0;

        for (i = 0; i < num_sends; i++)
        {
            for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
            {
                j_row = send_map_elmts[j];
                index = AT_offd_i[j_row];

                for (k = 0; k < AT_buf_i[j]; k++)
                {
                    if (data)
                    {
                        AT_offd_data[index] = AT_buf_data[counter];
                    }

                    AT_offd_j[index++] = AT_buf_j[counter++];
                }

                AT_offd_i[j_row] = index;
            }
        }

        for (i = num_cols; i > 0; i--)
        {
            AT_offd_i[i] = AT_offd_i[i - 1];
        }

        AT_offd_i[0] = 0;

        if (counter)
        {
            qsort0(AT_buf_j, 0, counter - 1);
            num_cols_offd_AT = 1;
            value = AT_buf_j[0];

            for (i = 1; i < counter; i++)
            {
                if (value < AT_buf_j[i])
                {
                    AT_buf_j[num_cols_offd_AT++] = AT_buf_j[i];
                    value = AT_buf_j[i];
                }
            }
        }

        if (num_cols_offd_AT)
        {
            col_map_offd_AT.resize(num_cols_offd_AT);
        }

        for (i = 0; i < num_cols_offd_AT; i++)
        {
            col_map_offd_AT[i] = AT_buf_j[i];
        }

        hypre_TFree(AT_buf_i);
        hypre_TFree(AT_buf_j);

        if (data)
        {
            hypre_TFree(AT_buf_data);
        }

        for (i = 0; i < counter; i++)
            AT_offd_j[i] = hypre_BinarySearch(col_map_offd_AT.data(), AT_offd_j[i],
                                              num_cols_offd_AT);
    }


#ifdef HYPRE_NO_GLOBAL_PARTITION
    row_starts_AT.resize(3);
    col_starts_AT.resize(3);

    for (i = 0; i < 2; i++)
    {
        row_starts_AT[i] = col_starts[i];
    }

    for (i = 0; i < 2; i++)
    {
        col_starts_AT[i] = row_starts[i];
    }

    row_starts_AT[2] = GlobalCols();
    col_starts_AT[2] = GlobalRows();

#else
    row_starts_AT.resize(num_procs + 1);
    col_starts_AT.resize(num_procs + 1);

    for (i = 0; i < num_procs + 1; i++)
    {
        row_starts_AT[i] = col_starts[i];
    }

    for (i = 0; i < num_procs + 1; i++)
    {
        col_starts_AT[i] = row_starts[i];
    }

#endif

    linalgcpp::SparseMatrix<double> AT_diag = diag_.Transpose();
    linalgcpp::SparseMatrix<double> AT_offd(std::move(AT_offd_i), std::move(AT_offd_j), std::move(AT_offd_data),
                                            num_cols, num_cols_offd_AT);

    return ParMatrix(comm, std::move(row_starts_AT), std::move(col_starts_AT),
                     std::move(AT_diag), std::move(AT_offd),
                     std::move(col_map_offd_AT));
}

HYPRE_Int hypre_ParCSRComputeL1Norms(ParMatrix& par_A,
                                     HYPRE_Int option,
                                     std::vector<HYPRE_Real>& l1_norm)
{
    hypre_ParCSRMatrix* A = par_A;
    HYPRE_Int i, j;
    HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

    hypre_CSRMatrix* A_diag = hypre_ParCSRMatrixDiag(A);
    HYPRE_Int* A_diag_I = hypre_CSRMatrixI(A_diag);
    HYPRE_Real* A_diag_data = hypre_CSRMatrixData(A_diag);

    hypre_CSRMatrix* A_offd = hypre_ParCSRMatrixOffd(A);
    HYPRE_Int* A_offd_I = hypre_CSRMatrixI(A_offd);
    HYPRE_Real* A_offd_data = hypre_CSRMatrixData(A_offd);
    HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

    HYPRE_Real diag;
    l1_norm.resize(num_rows, 0.0);

    if (option == 1)
    {
        for (i = 0; i < num_rows; i++)
        {
            l1_norm[i] = 0.0;
            {
                /* Add the l1 norm of the diag part of the ith row */
                for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
                {
                    l1_norm[i] += fabs(A_diag_data[j]);
                }

                /* Add the l1 norm of the offd part of the ith row */
                if (num_cols_offd)
                {
                    for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                    {
                        l1_norm[i] += fabs(A_offd_data[j]);
                    }
                }
            }
        }
    }
    else if (option == 2)
    {
        for (i = 0; i < num_rows; i++)
        {
            /* Add the diag element of the ith row */
            l1_norm[i] = fabs(A_diag_data[A_diag_I[i]]);
            {
                /* Add the l1 norm of the offd part of the ith row */
                if (num_cols_offd)
                {
                    for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                    {
                        l1_norm[i] += fabs(A_offd_data[j]);
                    }
                }
            }
        }
    }
    else if (option == 3)
    {
        for (i = 0; i < num_rows; i++)
        {
            l1_norm[i] = 0.0;

            for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
            {
                l1_norm[i] += A_diag_data[j] * A_diag_data[j];
            }

            if (num_cols_offd)
                for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                {
                    l1_norm[i] += A_offd_data[j] * A_offd_data[j];
                }
        }
    }
    else if (option == 4)
    {
        for (i = 0; i < num_rows; i++)
        {
            /* Add the diag element of the ith row */
            diag = l1_norm[i] = fabs(A_diag_data[A_diag_I[i]]);
            {
                /* Add the scaled l1 norm of the offd part of the ith row */
                if (num_cols_offd)
                {
                    for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                    {
                        l1_norm[i] += 0.5 * fabs(A_offd_data[j]);
                    }
                }
            }

            /* Truncate according to Remark 6.2 */
            if (l1_norm[i] <= 4.0 / 3.0 * diag)
            {
                l1_norm[i] = diag;
            }
        }
    }

    /* Handle negative definite matrices */
    for (i = 0; i < num_rows; i++)
        if (A_diag_data[A_diag_I[i]] < 0)
        {
            l1_norm[i] = -l1_norm[i];
        }

    for (i = 0; i < num_rows; i++)

        /* if (fabs(l1_norm[i]) < DBL_EPSILON) */
        if (fabs(l1_norm[i]) == 0.0)
        {
            hypre_error_in_arg(1);
            break;
        }

    return hypre_error_flag;
}



} //namespace parlinalgcpp
