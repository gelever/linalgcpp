/*! @file ParComPkg header */

#ifndef PARCOMMPKG_HPP__
#define PARCOMMPKG_HPP__

#include <vector>
#include <assert.h>
#include <algorithm>

#include "_hypre_parcsr_mv.h"

namespace parlinalgcpp
{

/*! @brief Hypre's Communication Package copied
    in C++ data structures.
*/

struct ParCommPkg
{
    /*! @brief Default Constructor */
    ParCommPkg() = default;

    /*! @brief Constructor given an existing hypre comm package */
    ParCommPkg(const hypre_ParCSRMatrix* A);

    /*! @brief Copy Constructor */
    ParCommPkg(const ParCommPkg& other) = default;

    /*! @brief Move Constructor */
    ParCommPkg(ParCommPkg&& other) = default;

    /*! @brief Assignment Constructor */
    ParCommPkg& operator=(ParCommPkg&& other) = default;

    /*! @brief Default Destructor */
    ~ParCommPkg() = default;

    MPI_Comm comm_;

    HYPRE_Int num_sends_;
    std::vector<HYPRE_Int> send_procs_; // num_sends
    std::vector<HYPRE_Int> send_map_starts_; // num_sends + 1
    std::vector<HYPRE_Int> send_map_elmts_; // send_map_starts_.end()

    HYPRE_Int num_recvs_;
    std::vector<HYPRE_Int> recv_procs_; // num_recvs
    std::vector<HYPRE_Int> recv_vec_starts_;  // num_recvs + 1
};

} // namespace parlinalgcpp

#endif // PARCOMMPKG_HPP__
