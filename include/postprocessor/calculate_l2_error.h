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
void calculate_L2_error(DoFHandler<dim> const                       &dof_handler,
                        Mapping<dim> const                          &mapping,
                        parallel::distributed::Vector<double> const &numerical_solution,
                        std::shared_ptr<Function<dim> >       analytical_solution,
                        double                                      &error,
                        bool                                        &relative_error,
                        unsigned int const                          additional_quadrature_points = 3)
{
  ConditionalOStream pcout(std::cout,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Vector<double> error_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
  Vector<double> solution_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
  VectorTools::integrate_difference (mapping,
                                     dof_handler,
                                     numerical_solution,
                                     *analytical_solution,
                                     error_norm_per_cell,
                                     QGauss<dim>(dof_handler.get_fe().degree + additional_quadrature_points),
                                     VectorTools::L2_norm);
  parallel::distributed::Vector<double> zero_solution;
  zero_solution.reinit(numerical_solution);
  VectorTools::integrate_difference (mapping,
                                     dof_handler,
                                     zero_solution,
                                     *analytical_solution,
                                     solution_norm_per_cell,
                                     QGauss<dim>(dof_handler.get_fe().degree + additional_quadrature_points),
                                     VectorTools::L2_norm);

  double error_norm = std::sqrt(Utilities::MPI::sum (error_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
  double solution_norm = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));

  if(solution_norm > 1.e-12)
  {
    error = error_norm/solution_norm;
    relative_error = true;
  }
  else
  {
    error = error_norm;
    relative_error = false;
  }
}

#endif /* INCLUDE_POSTPROCESSOR_CALCULATE_L2_ERROR_H_ */
