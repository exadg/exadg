/*
 * calculate_relative_l2_error.h
 *
 *  Created on: Mar 15, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_CALCULATE_L2_ERROR_H_
#define INCLUDE_POSTPROCESSOR_CALCULATE_L2_ERROR_H_

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

template<int dim>
double calculate_error(bool const                                  &relative_error,
                       DoFHandler<dim> const                       &dof_handler,
                       Mapping<dim> const                          &mapping,
                       parallel::distributed::Vector<double> const &numerical_solution,
                       std::shared_ptr<Function<dim> > const       analytical_solution,
                       double const                                &time,
                       VectorTools::NormType const                 &norm_type,
                       unsigned int const                          additional_quadrature_points = 3)
{
  double error = 1.0;
  analytical_solution->set_time(time);

  // calculate error norm
  Vector<double> error_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
  VectorTools::integrate_difference (mapping,
                                     dof_handler,
                                     numerical_solution,
                                     *analytical_solution,
                                     error_norm_per_cell,
                                     QGauss<dim>(dof_handler.get_fe().degree + additional_quadrature_points),
                                     norm_type);

  double error_norm = std::sqrt(Utilities::MPI::sum (error_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));

  if(relative_error == true)
  {
    // calculate solution norm
    Vector<double> solution_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
    parallel::distributed::Vector<double> zero_solution;
    zero_solution.reinit(numerical_solution);
    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       zero_solution,
                                       *analytical_solution,
                                       solution_norm_per_cell,
                                       QGauss<dim>(dof_handler.get_fe().degree + additional_quadrature_points),
                                       norm_type);

    double solution_norm = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));

    AssertThrow(solution_norm > 1.e-15, ExcMessage("Cannot compute relative error since absolute error tends to zero."));

    error = error_norm/solution_norm;
  }
  else // absolute error
  {
    error = error_norm;
  }

  return error;
}

#endif /* INCLUDE_POSTPROCESSOR_CALCULATE_L2_ERROR_H_ */
