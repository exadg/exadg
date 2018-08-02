#include "mg_coarse_ml_cg.h"

#ifdef DEAL_II_WITH_TRILINOS

using namespace dealii;

template <int DIM, typename Number>
MGCoarseMLCG<DIM, Number>::MGCoarseMLCG(
    const int level, const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_dg,
    const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_q,
    TrilinosWrappers::SparseMatrix &system_matrix)
    : MGCoarseMLWrapper<DIM, Number>(level, coarse_matrix_q, system_matrix),
      coarse_matrix_dg(coarse_matrix_dg),
      transfer(this->coarse_matrix_dg.get_data(),
               this->coarse_matrix.get_data(), level, this->degree) {}

template <int DIM, typename Number>
void MGCoarseMLCG<DIM, Number>::vmult_pre(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {

  this->transfer.toCG(dst, src);
}

template <int DIM, typename Number>
void MGCoarseMLCG<DIM, Number>::vmult_post(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {

  this->transfer.toDG(dst, src);
}

#include "mg_coarse_ml_cg.hpp"

#endif