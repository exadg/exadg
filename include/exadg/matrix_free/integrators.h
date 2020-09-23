#ifndef INCLUDE_EXADG_MATRIX_FREE_INTEGRATORS_H_
#define INCLUDE_EXADG_MATRIX_FREE_INTEGRATORS_H_

// deal.II
#include <deal.II/base/config.h>
#include <deal.II/matrix_free/fe_evaluation.h>

DEAL_II_NAMESPACE_OPEN

template<int dim,
         int n_components,
         typename Number,
         typename VectorizedArrayType = VectorizedArray<Number>>
using CellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

template<int dim,
         int n_components,
         typename Number,
         typename VectorizedArrayType = VectorizedArray<Number>>
using FaceIntegrator = FEFaceEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

DEAL_II_NAMESPACE_CLOSE

#endif
