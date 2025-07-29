/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#ifndef EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_
#define EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_

// deal.II
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/inverse_mass_parameters.h>

namespace ExaDG
{
namespace GridToGridProjection
{
// Parameters controlling the grid-to-grid projection.
template<int dim>
struct GridToGridProjectionData
{
  GridToGridProjectionData()
    : rpe_data(),
      solver_data(),
      preconditioner(PreconditionerMass::PointJacobi),
      inverse_mass_type(InverseMassType::MatrixfreeOperator),
      additional_quadrature_points(1),
      is_test(false)
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "RPE tolerance", rpe_data.tolerance);
    print_parameter(pcout, "RPE enforce unique mapping", rpe_data.enforce_unique_mapping);
    print_parameter(pcout, "RPE rtree level", rpe_data.rtree_level);
  }

  typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData::AdditionalData
                     rpe_data;
  SolverData         solver_data;
  PreconditionerMass preconditioner;
  InverseMassType    inverse_mass_type;

  // Number of additional integration points used for sampling the source grid.
  // The default `additional_quadrature_points = 1` considers `fe_degree + 1` quadrature points in
  // 1D using the `fe_degree` of the target grid's finite element.
  unsigned int additional_quadrature_points;

  // Toggle iteration count and timing output.
  bool is_test;
};

/**
 * Utility function to collect integration points in the particular sequence they are encountered
 * in.
 */
template<int dim, int n_components, typename Number>
std::vector<dealii::Point<dim>>
collect_integration_points(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index);

/**
 * Utility function to compute the right hand side of a projection (mass matrix solve)
 * with values given in integration points in the particular sequence they are encountered in.
 */
template<int dim, int n_components, typename Number, typename VectorType>
VectorType
assemble_projection_rhs(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  std::vector<
    typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const &
                     values_source_in_q_points_target,
  unsigned int const dof_index,
  unsigned int const quad_index);

/**
 * Utilitiy function to project vectors from a source to a target triangulation via
 * matrix-free mass operator evaluation and preconditioned CG solver.
 */
template<int dim, typename Number, int n_components, typename VectorType>
void
project_vectors(
  std::vector<VectorType *> const &                                        source_vectors,
  dealii::DoFHandler<dim> const &                                          source_dof_handler,
  std::shared_ptr<dealii::Mapping<dim> const> const &                      source_mapping,
  std::vector<VectorType *> const &                                        target_vectors,
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & target_matrix_free,
  dealii::AffineConstraints<Number> const &                                constraints,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index,
  GridToGridProjectionData<dim> const &                                    data);

/**
 * Utility function to perform matrix-free grid-to-grid projection. We assume we only have a single
 * `dealii::FiniteElement` per `dealii::DoFHandler`. This function creates a `MatrixFree` object.
 */
template<int dim, typename Number, typename VectorType>
void
grid_to_grid_projection(
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping,
  GridToGridProjectionData<dim> const &                data);

/**
 * Same as the function above, but relies on a suitable `MatrixFree` object input argument.
 * We assume that the `MatrixFree` object
 * */
template<int dim, typename Number, typename VectorType>
void
do_grid_to_grid_projection(
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping,
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  GridToGridProjectionData<dim> const &                                    data);

} // namespace GridToGridProjection
} // namespace ExaDG

#endif /* EXADG_OPERATORS_SOLUTION_PROJECTION_BETWEEN_TRIANGULATIONS_H_ */
