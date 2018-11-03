/*! @file */

#ifndef PARPRECOND_HPP__
#define PARPRECOND_HPP__

#include "parmatrix.hpp"

namespace linalgcpp
{

class ParBlockDiagComp : public ParOperator
{
    public:
        ParBlockDiagComp() = default;
        ParBlockDiagComp(const ParMatrix& A, const ParMatrix& agg_vertex);

        ~ParBlockDiagComp() = default;

        /*! @brief Apply the action of the solver
            @param input input vector x
            @param output output vector y
        */
        void Mult(const linalgcpp::VectorView<double>& input,
                 linalgcpp::VectorView<double> output) const;

    private:
        ParMatrix MakeRedistributor(const ParMatrix& agg_vertex);

        ParMatrix redist_;

        BlockOperator block_op_;
        std::vector<std::unique_ptr<Operator>> solvers_;

        mutable Vector<double> sol_r_;
        mutable Vector<double> rhs_r_;
};

} //namespace linalgcpp

#endif // PARPRECOND_HPP
