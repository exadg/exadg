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

#ifndef INCLUDE_EXADG_OPERATORS_GRID_RELATED_TIME_STEP_RESTRICTIONS_H_
#define INCLUDE_EXADG_OPERATORS_GRID_RELATED_TIME_STEP_RESTRICTIONS_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/grid/calculate_characteristic_element_length.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/time_integration/enum_types.h>

namespace ExaDG
{
/*
 * This function calculates the time step such that the spatial and temporal errors are of the same
 * order of magnitude:
 *
 *    spatial error: e = C_h * h^{p+1}
 *    temporal error: e = C_dt * dt^{n} (n: order of time integration method)
 *
 * where h is a mesh size defined globally for the whole mesh, e.g. the global minimum vertex
 * distance.
 */
inline double
calculate_time_step_max_efficiency(double const       h,
                                   unsigned int const fe_degree,
                                   unsigned int const order_time_integration)
{
  double const exponent  = (double)(fe_degree + 1) / order_time_integration;
  double const time_step = std::pow(h, exponent);

  return time_step;
}

/*
 * This function calculates the time step according to the formula
 *
 *    diffusion_number/k^{exponent_fe_degree} = diffusivity * time_step / h²
 */
inline double
calculate_const_time_step_diff(double const       diffusivity,
                               double const       global_min_cell_diameter,
                               unsigned int const fe_degree,
                               double const       exponent_fe_degree = 3.0)
{
  double const time_step =
    pow(global_min_cell_diameter, 2.0) / pow(fe_degree, exponent_fe_degree) / diffusivity;

  return time_step;
}

/*
 * This function calculates the maximum velocity for a given velocity field (which is known
 * analytically). The maximum value is defined as the maximum velocity at the cell center.
 */
template<int dim>
inline double
calculate_max_velocity(dealii::Triangulation<dim> const &     triangulation,
                       std::shared_ptr<dealii::Function<dim>> velocity,
                       double const                           time,
                       MPI_Comm const &                       mpi_comm)
{
  double U = 0.0, max_U = std::numeric_limits<double>::min();

  dealii::Tensor<1, dim, double> vel;
  velocity->set_time(time);

  for(auto const & cell : triangulation.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      // calculate maximum velocity
      dealii::Point<dim> point = cell->center();

      for(unsigned int d = 0; d < dim; ++d)
        vel[d] = velocity->value(point, d);

      U = vel.norm();
      if(U > max_U)
        max_U = U;
    }
  }
  double const global_max_U = dealii::Utilities::MPI::max(max_U, mpi_comm);

  return global_max_U;
}

/*
 * Calculate time step size according to local CFL criterion where the velocity field is a
 * prescribed analytical function. The computed time step size corresponds to CFL = 1.0.
 *
 * The underlying CFL condition reads
 *
 * 	cfl/k^{exponent_fe_degree} = || U || * time_step / h
 */
template<int dim, typename value_type>
inline double
calculate_time_step_cfl_local(dealii::MatrixFree<dim, value_type> const &  data,
                              unsigned int const                           dof_index,
                              unsigned int const                           quad_index,
                              std::shared_ptr<dealii::Function<dim>> const velocity,
                              double const                                 time,
                              unsigned int const                           degree,
                              double const                                 exponent_fe_degree,
                              CFLConditionType const                       cfl_condition_type,
                              MPI_Comm const &                             mpi_comm)
{
  CellIntegrator<dim, dim, value_type> fe_eval(data, dof_index, quad_index);

  double new_time_step = std::numeric_limits<double>::max();

  double const cfl_p = 1.0 / pow(degree, exponent_fe_degree);

  // loop over cells of processor
  for(unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
    dealii::VectorizedArray<value_type> delta_t_cell =
      dealii::make_vectorized_array<value_type>(std::numeric_limits<value_type>::max());

    fe_eval.reinit(cell);

    // loop over quadrature points
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      dealii::Point<dim, dealii::VectorizedArray<value_type>> q_point = fe_eval.quadrature_point(q);

      dealii::Tensor<1, dim, dealii::VectorizedArray<value_type>> u_x =
        FunctionEvaluator<1, dim, value_type>::value(*velocity, q_point, time);
      dealii::Tensor<2, dim, dealii::VectorizedArray<value_type>> invJ =
        fe_eval.inverse_jacobian(q);
      invJ                                                              = transpose(invJ);
      dealii::Tensor<1, dim, dealii::VectorizedArray<value_type>> ut_xi = invJ * u_x;

      if(cfl_condition_type == CFLConditionType::VelocityNorm)
      {
        delta_t_cell = std::min(delta_t_cell, cfl_p / ut_xi.norm());
      }
      else if(cfl_condition_type == CFLConditionType::VelocityComponents)
      {
        for(unsigned int d = 0; d < dim; ++d)
          delta_t_cell = std::min(delta_t_cell, cfl_p / (std::abs(ut_xi[d])));
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }

    // loop over vectorized array
    double dt = std::numeric_limits<double>::max();
    for(unsigned int v = 0; v < dealii::VectorizedArray<value_type>::size(); ++v)
    {
      dt = std::min(dt, (double)delta_t_cell[v]);
    }

    new_time_step = std::min(new_time_step, dt);
  }

  // find minimum over all processors
  new_time_step = dealii::Utilities::MPI::min(new_time_step, mpi_comm);

  return new_time_step;
}

/*
 * Calculate time step size according to local CFL criterion where the velocity field is a numerical
 * solution field. The computed time step size corresponds to CFL = 1.0.
 */
template<int dim, typename value_type>
inline double
calculate_time_step_cfl_local(
  dealii::MatrixFree<dim, value_type> const &                    data,
  unsigned int const                                             dof_index,
  unsigned int const                                             quad_index,
  dealii::LinearAlgebra::distributed::Vector<value_type> const & velocity,
  unsigned int const                                             degree,
  double const                                                   exponent_fe_degree,
  CFLConditionType const                                         cfl_condition_type,
  MPI_Comm const &                                               mpi_comm)
{
  CellIntegrator<dim, dim, value_type> fe_eval(data, dof_index, quad_index);

  double new_time_step = std::numeric_limits<double>::max();

  double const cfl_p = 1.0 / pow(degree, exponent_fe_degree);

  // loop over cells of processor
  for(unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
    dealii::VectorizedArray<value_type> delta_t_cell =
      dealii::make_vectorized_array<value_type>(std::numeric_limits<value_type>::max());

    dealii::Tensor<2, dim, dealii::VectorizedArray<value_type>> invJ;
    dealii::Tensor<1, dim, dealii::VectorizedArray<value_type>> u_x;
    dealii::Tensor<1, dim, dealii::VectorizedArray<value_type>> ut_xi;

    fe_eval.reinit(cell);
    fe_eval.read_dof_values(velocity);
    fe_eval.evaluate(dealii::EvaluationFlags::values);

    // loop over quadrature points
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      u_x   = fe_eval.get_value(q);
      invJ  = fe_eval.inverse_jacobian(q);
      invJ  = transpose(invJ);
      ut_xi = invJ * u_x;

      if(cfl_condition_type == CFLConditionType::VelocityNorm)
      {
        delta_t_cell = std::min(delta_t_cell, cfl_p / ut_xi.norm());
      }
      else if(cfl_condition_type == CFLConditionType::VelocityComponents)
      {
        for(unsigned int d = 0; d < dim; ++d)
          delta_t_cell = std::min(delta_t_cell, cfl_p / (std::abs(ut_xi[d])));
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }

    // loop over vectorized array
    double dt = std::numeric_limits<double>::max();
    for(unsigned int v = 0; v < dealii::VectorizedArray<value_type>::size(); ++v)
    {
      dt = std::min(dt, (double)delta_t_cell[v]);
    }

    new_time_step = std::min(new_time_step, dt);
  }

  // find minimum over all processors
  new_time_step = dealii::Utilities::MPI::min(new_time_step, mpi_comm);

  // Cut time step size after, e.g., 4 digits of accuracy in order to make sure that there is no
  // drift in the time step size depending on the number of processors when using adaptive time
  // stepping. This effect can occur since the velocity field and the time step size are coupled
  // (there is some form of feedback loop in case of adaptive time stepping, i.e., a minor change
  // in the time step size due to round-off errors implies that the velocity field is evaluated
  // at a slightly different time in the next time step and so on). This way, it can be ensured
  // that the sequence of time step sizes is exactly reproducible with the results being
  // independent of the number of processors, which is important for code verification in a
  // parallel setting.
  new_time_step = dealii::Utilities::truncate_to_n_digits(new_time_step, 4);

  return new_time_step;
}

/*
 * this function computes the actual CFL number in each cell given a global time step size
 * (that holds for all cells)
 */

template<int dim, typename value_type>
void
calculate_cfl(dealii::LinearAlgebra::distributed::Vector<value_type> &       cfl,
              dealii::Triangulation<dim> const &                             triangulation,
              dealii::MatrixFree<dim, value_type> const &                    data,
              unsigned int const                                             dof_index,
              unsigned int const                                             quad_index,
              dealii::LinearAlgebra::distributed::Vector<value_type> const & velocity,
              double const                                                   time_step,
              unsigned int const                                             degree,
              double const                                                   exponent_fe_degree)
{
  CellIntegrator<dim, dim, value_type> fe_eval(data, dof_index, quad_index);

  unsigned int const n_active_cells = triangulation.n_active_cells();

  cfl.reinit(n_active_cells);

  for(unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(velocity);
    fe_eval.evaluate(dealii::EvaluationFlags::values);

    dealii::VectorizedArray<value_type> u_va =
      dealii::make_vectorized_array<value_type>(std::numeric_limits<value_type>::min());

    dealii::Tensor<2, dim, dealii::VectorizedArray<value_type>> invJ;
    dealii::Tensor<1, dim, dealii::VectorizedArray<value_type>> u_x;
    dealii::Tensor<1, dim, dealii::VectorizedArray<value_type>> ut_xi;

    // loop over quadrature points
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      u_x   = fe_eval.get_value(q);
      invJ  = fe_eval.inverse_jacobian(q);
      invJ  = transpose(invJ);
      ut_xi = invJ * u_x;

      u_va = std::max(u_va, ut_xi.norm());
    }

    // loop over vectorized array
    for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    {
      cfl[data.get_cell_iterator(cell, v)->active_cell_index()] =
        u_va[v] * time_step * pow(degree, exponent_fe_degree);
    }
  }
}

} // namespace ExaDG


#endif /* INCLUDE_EXADG_OPERATORS_GRID_RELATED_TIME_STEP_RESTRICTIONS_H_ */
