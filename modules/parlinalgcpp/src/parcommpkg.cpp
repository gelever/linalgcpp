#include "parcommpkg.hpp"

namespace linalgcpp
{

ParCommPkg::ParCommPkg(const hypre_ParCSRMatrix* A)
    : comm_(A->comm)
{
    linalgcpp_assert(A != NULL);

    const hypre_ParCSRCommPkg* comm_pkg = A->comm_pkg;

    linalgcpp_assert(comm_pkg != NULL);

    num_sends_ = comm_pkg->num_sends;

    send_procs_.resize(num_sends_);
    std::copy_n(comm_pkg->send_procs, num_sends_, std::begin(send_procs_));

    send_map_starts_.resize(num_sends_ + 1);
    std::copy_n(comm_pkg->send_map_starts, num_sends_ + 1, std::begin(send_map_starts_));

    send_map_elmts_.resize(send_map_starts_.back());
    std::copy_n(comm_pkg->send_map_elmts, send_map_starts_.back(), std::begin(send_map_elmts_));

    num_recvs_ = comm_pkg->num_recvs;

    recv_procs_.resize(num_recvs_);
    std::copy_n(comm_pkg->recv_procs, num_recvs_, std::begin(recv_procs_));

    recv_vec_starts_.resize(num_recvs_ + 1);
    std::copy_n(comm_pkg->recv_vec_starts, num_recvs_ + 1, std::begin(recv_vec_starts_));
}

} // namespace linalgcpp
