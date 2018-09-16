/*! @file Rewrite of hypre operations to use std::vector for memory instead of malloc */
/*  BHEADER**********************************************************************
    Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
    Produced at the Lawrence Livermore National Laboratory.
    This file is part of HYPRE.  See file COPYRIGHT for details.

    HYPRE is free software; you can redistribute it and/or modify it under the
    terms of the GNU Lesser General Public License (as published by the Free
    Software Foundation) version 2.1 dated February 1999.

    $Revision$
 ***********************************************************************EHEADER*/
#ifndef HYPRE_OPS_HPP__
#define HYPRE_OPS_HPP__

#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "parmatrix.hpp"

namespace parlinalgcpp
{

/// Modified from hypre
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
);

/// Modified from hypre
HYPRE_Int hypre_ParCSRComputeL1Norms(ParMatrix& par_A, HYPRE_Int option,
                                     std::vector<HYPRE_Real>& l1_norms);

/// Wrapper for PCG create to match others
/** hypre-2.10.b does not use the comm,
    so this should be safe
*/

inline
HYPRE_Int HYPRE_ParCSRPCGCreate(HYPRE_Solver* solver, MPI_Comm comm)
{
    return HYPRE_ParCSRPCGCreate(comm, solver);
}

inline
HYPRE_Int HYPRE_ParCSRPCGCreate(HYPRE_Solver* solver)
{
    return HYPRE_ParCSRPCGCreate(solver, MPI_COMM_WORLD);
}

} //namespace parlinalgcpp

#endif // HYPRE_OPS_HPP__
