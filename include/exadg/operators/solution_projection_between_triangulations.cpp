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

// C/C++
#include <algorithm>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/operators/quadrature.h>
#include <exadg/operators/solution_projection_between_triangulations.h>
#include <exadg/postprocessor/write_output.h>
#include <exadg/solvers_and_preconditioners/preconditioners/additive_schwarz_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace GridToGridProjection
{
template<int dim, int n_components, typename Number>
std::vector<dealii::Point<dim>>
collect_integration_points(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index)
{
  CellIntegrator<dim, n_components, Number> fe_eval(matrix_free, dof_index, quad_index);

  // Conservative estimate for the number of points.
  std::vector<dealii::Point<dim>> integration_points;
  integration_points.reserve(
    matrix_free.get_dof_handler(dof_index).get_triangulation().n_active_cells() *
    fe_eval.n_q_points);

  for(unsigned int cell_batch_idx = 0; cell_batch_idx < matrix_free.n_cell_batches();
      ++cell_batch_idx)
  {
    fe_eval.reinit(cell_batch_idx);
    for(const unsigned int q : fe_eval.quadrature_point_indices())
    {
      dealii::Point<dim, dealii::VectorizedArray<Number>> const cell_batch_points =
        fe_eval.quadrature_point(q);
      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        dealii::Point<dim> p;
        for(unsigned int d = 0; d < dim; ++d)
        {
          p[d] = cell_batch_points[d][i];
        }
        integration_points.push_back(p);
      }
    }
  }

  return integration_points;
}

template<int dim, int n_components, typename Number, typename VectorType>
VectorType
assemble_projection_rhs(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  CellIntegrator<dim, n_components, Number> &                              fe_eval,
  std::vector<
    typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const &
                     values_source_in_q_points_target,
  unsigned int const dof_index)
{
  VectorType system_rhs;
  matrix_free.initialize_dof_vector(system_rhs, dof_index);

  unsigned int idx_q_point = 0;

  for(unsigned int cell_batch_idx = 0; cell_batch_idx < matrix_free.n_cell_batches();
      ++cell_batch_idx)
  {
    fe_eval.reinit(cell_batch_idx);
    for(unsigned int const q : fe_eval.quadrature_point_indices())
    {
      dealii::Tensor<1, n_components, dealii::VectorizedArray<Number>> tmp;

      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type const
          values = values_source_in_q_points_target[idx_q_point];

        // Increment index into `values_source_in_q_points_target`, meaning that the sequence of
        // function values in integration points need to match the particular sequence here.
        ++idx_q_point;

        if constexpr(n_components == 1)
        {
          tmp[0][i] = values;
        }
        else
        {
          for(unsigned int c = 0; c < n_components; ++c)
          {
            tmp[c][i] = values[c];
          }
        }
      }

      fe_eval.submit_value(tmp, q);
    }
    fe_eval.integrate(dealii::EvaluationFlags::values);
    fe_eval.distribute_local_to_global(system_rhs);
  }
  system_rhs.compress(dealii::VectorOperation::add);

  return system_rhs;
}

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
  GridToGridProjectionData<dim> const &                                    data)
{
  // Setup operator and preconditioner outside of the loop since the operator remains unchanged.
  MassOperatorData<dim> mass_operator_data;
  mass_operator_data.dof_index  = dof_index;
  mass_operator_data.quad_index = quad_index;

  using MassOperatorType = MassOperator<dim, n_components, Number>;
  MassOperatorType mass_operator;
  mass_operator.initialize(target_matrix_free, constraints, mass_operator_data);

  // Setup preconditioner.
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;
  if(data.preconditioner == Preconditioner::None)
  {
    AssertThrow(not data.solver_data.use_preconditioner,
                dealii::ExcMessage("Conflicting settings in `Preconditioner` and CG solver data."));
    // dummy `JacobiPreconditioner` never used.
    preconditioner = std::make_shared<JacobiPreconditioner<MassOperatorType>>(
      mass_operator, false /* initialize_preconditioner */);
  }
  else if(data.preconditioner == Preconditioner::PointJacobi)
  {
    preconditioner = std::make_shared<JacobiPreconditioner<MassOperatorType>>(
      mass_operator, true /* initialize_preconditioner */);
  }
  else if(data.preconditioner == Preconditioner::InverseMass)
  {
    InverseMassOperatorData inverse_mass_operator_data;
    inverse_mass_operator_data.dof_index  = dof_index;
    inverse_mass_operator_data.quad_index = quad_index;
    ElementType element_type =
      get_element_type(target_matrix_free.get_dof_handler(dof_index).get_triangulation());
    inverse_mass_operator_data.parameters.implementation_type =
      element_type == ElementType::Hypercube and data.additional_quadrature_points == 1 ?
        InverseMassType::MatrixfreeOperator :
        InverseMassType::ElementwiseKrylovSolver;

    preconditioner = std::make_shared<InverseMassPreconditioner<dim, n_components, Number>>(
      target_matrix_free, inverse_mass_operator_data);
  }
  else if(data.preconditioner == Preconditioner::AMG)
  {
    AMGData amg_data;
#ifdef DEAL_II_WITH_TRILINOS
    amg_data.ml_data.smoother_type   = "Chebyshev";
    amg_data.ml_data.smoother_sweeps = 6;
#endif
    preconditioner =
      std::make_shared<PreconditionerAMG<MassOperatorType, Number>>(mass_operator,
                                                                    true /* initialize */,
                                                                    amg_data);
  }
  else if(data.preconditioner == Preconditioner::AdditiveSchwarz)
  {
    preconditioner =
      std::make_shared<AdditiveSchwarzPreconditioner<MassOperatorType>>(mass_operator,
                                                                        true /* initialize */);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("This type of preconditioner is not implemented."));
  }

  // Setup global CG solver.
  Krylov::SolverCG<MassOperatorType, PreconditionerBase<Number>, VectorType> solver(
    mass_operator, *preconditioner, data.solver_data);

  dealii::Utilities::MPI::RemotePointEvaluation<dim> rpe_source(data.rpe_data);

  // The sequence of integration points follows from the sequence of points as encountered during
  // cell batch loop.
  std::vector<dealii::Point<dim>> integration_points_target =
    collect_integration_points<dim, n_components, Number>(target_matrix_free,
                                                          dof_index,
                                                          quad_index);

  rpe_source.reinit(integration_points_target,
                    source_dof_handler.get_triangulation(),
                    *source_mapping);

  if(not rpe_source.all_points_found())
  {
    write_points_in_dummy_triangulation(
      integration_points_target, "./", "all_points", 0, source_dof_handler.get_communicator());

    std::vector<dealii::Point<dim>> points_not_found;
    points_not_found.reserve(integration_points_target.size());
    for(unsigned int i = 0; i < integration_points_target.size(); ++i)
    {
      if(not rpe_source.point_found(i))
      {
        points_not_found.push_back(integration_points_target[i]);
      }
    }

    write_points_in_dummy_triangulation(
      points_not_found, "./", "points_not_found", 0, source_dof_handler.get_communicator());

    AssertThrow(rpe_source.all_points_found(),
                dealii::ExcMessage(
                  "Could not interpolate source grid vector in target grid. "
                  "Points exported to `./all_points.pvtu` and `./points_not_found.pvtu`"));
  }

  CellIntegrator<dim, n_components, Number> fe_eval(target_matrix_free, dof_index, quad_index);

  // Loop over vectors and project.
  for(unsigned int i = 0; i < target_vectors.size(); ++i)
  {
    // Evaluate the source vector at the target integration points.
    VectorType const & source_vector = *source_vectors.at(i);
    source_vector.update_ghost_values();

    std::vector<
      typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const
      values_source_in_q_points_target = dealii::VectorTools::point_values<n_components>(
        rpe_source, source_dof_handler, source_vector, dealii::VectorTools::EvaluationFlags::avg);

    // Assemble right hand side vector for the projection.
    VectorType system_rhs = assemble_projection_rhs<dim, n_components, Number, VectorType>(
      target_matrix_free, fe_eval, values_source_in_q_points_target, dof_index);

    // Solve linear system starting from zero initial guess.
    VectorType sol;
    sol.reinit(system_rhs, false /* omit_zeroing_entries */);

    dealii::Timer              timer;
    dealii::ConditionalOStream pcout(std::cout,
                                     (dealii::Utilities::MPI::this_mpi_process(
                                        source_dof_handler.get_communicator()) == 0));

    timer.restart();

    unsigned int n_iter = solver.solve(sol, system_rhs);

    if(not data.is_test)
    {
      print_solver_info_linear(pcout, n_iter, timer.wall_time());
    }

    // Copy solution to target vector.
    *target_vectors[i] = sol;
  }
}

template<int dim, typename VectorType>
void
grid_to_grid_projection(
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  source_mapping,
  std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping,
  GridToGridProjectionData<dim> const &                data)
{
  // Check input dimensions.
  AssertThrow(source_vectors_per_dof_handler.size() == source_dof_handlers.size(),
              dealii::ExcMessage("First dimension of source vector of vectors "
                                 "has to match source DoFHandler count."));
  AssertThrow(target_vectors_per_dof_handler.size() == target_dof_handlers.size(),
              dealii::ExcMessage("First dimension of target vector of vectors "
                                 "has to match target DoFHandler count."));
  AssertThrow(source_dof_handlers.size() == target_dof_handlers.size(),
              dealii::ExcMessage("Target and source DoFHandler counts have to match"));
  AssertThrow(source_vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("Vector of source vectors empty."));
  for(unsigned int i = 0; i < source_vectors_per_dof_handler.size(); ++i)
  {
    AssertThrow(source_vectors_per_dof_handler[i].size() ==
                  target_vectors_per_dof_handler.at(i).size(),
                dealii::ExcMessage("Vectors of source and target vectors need to have same size."));
  }

  // Setup a single `dealii::MatrixFree` object with multiple `dealii::DoFHandler`s.
  using Number = typename VectorType::value_type;
  MatrixFreeData<dim, Number> matrix_free_data;

  MappingFlags mapping_flags;
  mapping_flags.cells =
    dealii::update_quadrature_points | dealii::update_values | dealii::update_JxW_values;
  matrix_free_data.append_mapping_flags(mapping_flags);

  dealii::AffineConstraints<Number> empty_constraints;
  empty_constraints.clear();
  empty_constraints.close();
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    matrix_free_data.insert_dof_handler(target_dof_handlers[i], std::to_string(i));
    matrix_free_data.insert_constraint(&empty_constraints, std::to_string(i));

    ElementType element_type = get_element_type(target_dof_handlers[i]->get_triangulation());

    std::shared_ptr<dealii::Quadrature<dim>> quadrature = create_quadrature<dim>(
      element_type, target_dof_handlers[i]->get_fe().degree + data.additional_quadrature_points);

    matrix_free_data.insert_quadrature(*quadrature, std::to_string(i));
  }

  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> matrix_free;
  matrix_free.reinit(*target_mapping,
                     matrix_free_data.get_dof_handler_vector(),
                     matrix_free_data.get_constraint_vector(),
                     matrix_free_data.get_quadrature_vector(),
                     matrix_free_data.data);

  // Project vectors per `dealii::DoFHandler`.
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    unsigned int const n_components = target_dof_handlers[i]->get_fe().n_components();
    if(n_components == 1)
    {
      project_vectors<dim, Number, 1 /* n_components */, VectorType>(
        source_vectors_per_dof_handler.at(i),
        *source_dof_handlers.at(i),
        source_mapping,
        target_vectors_per_dof_handler.at(i),
        matrix_free,
        empty_constraints,
        i /* dof_index */,
        i /* quad_index */,
        data);
    }
    else if(n_components == dim)
    {
      project_vectors<dim, Number, dim /* n_components */, VectorType>(
        source_vectors_per_dof_handler.at(i),
        *source_dof_handlers.at(i),
        source_mapping,
        target_vectors_per_dof_handler.at(i),
        matrix_free,
        empty_constraints,
        i /* dof_index */,
        i /* quad_index */,
        data);
    }
    else if(n_components == dim + 2)
    {
      project_vectors<dim, Number, dim + 2 /* n_components */, VectorType>(
        source_vectors_per_dof_handler.at(i),
        *source_dof_handlers.at(i),
        source_mapping,
        target_vectors_per_dof_handler.at(i),
        matrix_free,
        empty_constraints,
        i /* dof_index */,
        i /* quad_index */,
        data);
    }
    else
    {
      AssertThrow(n_components == 1 or n_components == dim,
                  dealii::ExcMessage("The requested number of components is not"
                                     "supported in `grid_to_grid_projection()`."));
    }
  }
}

} // namespace GridToGridProjection
} // namespace ExaDG
