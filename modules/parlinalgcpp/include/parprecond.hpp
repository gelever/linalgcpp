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
        ParBlockDiagComp(const ParMatrix& A, const ParMatrix& agg_vertex, int num_steps=1);

        ~ParBlockDiagComp() = default;

        using Operator::Mult;

        /*! @brief Apply the action of the solver
            @param input input vector x
            @param output output vector y
        */
        void Mult(const linalgcpp::VectorView<double>& input,
                 linalgcpp::VectorView<double> output) const;

        double Anorm(const linalgcpp::VectorView<double>& x) const;

    private:
        ParMatrix MakeRedistributer(const ParMatrix& agg_vertex);

        ParMatrix A_r_;
        ParMatrix redist_;

        BlockDiagOperator block_op_;
        std::vector<std::unique_ptr<Operator>> solvers_;

        int num_steps_;

        mutable Vector<double> x_;
        mutable Vector<double> b_;
};

} //namespace linalgcpp

#endif // PARPRECOND_HPP
