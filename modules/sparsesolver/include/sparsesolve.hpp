/*! @file */

#ifndef SPARSESOLVE_HPP__
#define SPARSESOLVE_HPP__

#include "umfpack.h"
#include "linalgcpp.hpp"

namespace linalgcpp
{

class SparseSolver : public linalgcpp::Operator
{
    public:
        SparseSolver() : numeric_(nullptr) {}
        explicit SparseSolver(linalgcpp::SparseMatrix<double> A);

        /// @warning Avoid copy since this will have to refactor A
        SparseSolver(const SparseSolver& other) noexcept;

        SparseSolver(SparseSolver&& other) noexcept;
        SparseSolver& operator=(SparseSolver other) noexcept;

        ~SparseSolver();

        friend void swap(SparseSolver& lhs, SparseSolver& rhs) noexcept;

        void Mult(const linalgcpp::VectorView<double>& input,
                  linalgcpp::VectorView<double> output) const override;
        void MultAT(const linalgcpp::VectorView<double>& input,
                  linalgcpp::VectorView<double> output) const override;

        using linalgcpp::Operator::Mult;

    private:
        void Init();

        linalgcpp::SparseMatrix<double> A_;
        void* numeric_;

        std::vector<double> control_;
        mutable std::vector<double> info_;
};

} // namespace linalgcpp

#endif // SPARSESOLVE_HPP__
