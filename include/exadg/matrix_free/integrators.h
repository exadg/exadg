/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

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
