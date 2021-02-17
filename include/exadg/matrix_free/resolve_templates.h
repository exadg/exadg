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

#include <exadg/configuration/config.h>

#define FE_EVAL_FACTORY_DEGREE_MAX EXADG_DEGREE_MAX

#include <deal.II/matrix_free/evaluation_template_factory.templates.h>

DEAL_II_NAMESPACE_OPEN

template struct dealii::internal::FEEvaluationFactory<deal_II_dimension,
                                                      deal_II_scalar_vectorized::value_type,
                                                      deal_II_scalar_vectorized>;

template struct dealii::internal::FEFaceEvaluationFactory<deal_II_dimension,
                                                          deal_II_scalar_vectorized::value_type,
                                                          deal_II_scalar_vectorized>;

template struct dealii::internal::CellwiseInverseMassFactory<deal_II_dimension,
                                                             deal_II_scalar_vectorized::value_type,
                                                             deal_II_scalar_vectorized>;

DEAL_II_NAMESPACE_CLOSE
