#include "mg_coarse_ml_dg.h"

using namespace dealii;

template <int DIM, typename Number>
MGCoarseMLDG<DIM, Number>::MGCoarseMLDG(
    const int level, const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix,
    const MatrixOperatorBase & /*coarse_matrix*/,
    TrilinosWrappers::SparseMatrix &system_matrix)
    : MGCoarseMLWrapper<DIM, Number>(level, coarse_matrix, system_matrix) {}

template <int DIM, typename Number>
void MGCoarseMLDG<DIM, Number>::vmult_pre(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {
  dst = src;
}

template <int DIM, typename Number>
void MGCoarseMLDG<DIM, Number>::vmult_post(
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src) {
  dst = src;
}

#include "mg_coarse_ml_dg.hpp"