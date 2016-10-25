/*
 * ErrorCalculationNavierStokes.h
 *
 *  Created on: Oct 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_ERRORCALCULATIONNAVIERSTOKES_H_
#define INCLUDE_ERRORCALCULATIONNAVIERSTOKES_H_

#include <deal.II/numerics/vector_tools.h>


template<int dim>
void calculate_L2_error(DoFHandler<dim> const                       &dof_handler,
                        Mapping<dim> const                          &mapping,
                        parallel::distributed::Vector<double> const &numerical_solution,
                        std_cxx11::shared_ptr<Function<dim> >       analytical_solution,
                        double                                      &error,
                        bool                                        &relative_error)
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
                                     QGauss<dim>(dof_handler.get_fe().degree+4),//(fe().degree+2),
                                     VectorTools::L2_norm);
  parallel::distributed::Vector<double> zero_solution;
  zero_solution.reinit(numerical_solution);
  VectorTools::integrate_difference (mapping,
                                     dof_handler,
                                     zero_solution,
                                     *analytical_solution,
                                     solution_norm_per_cell,
                                     QGauss<dim>(dof_handler.get_fe().degree+4), //(fe().degree+2),
                                     VectorTools::L2_norm);
  double error_norm = std::sqrt(Utilities::MPI::sum (error_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
  double solution_norm = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell.norm_sqr(), MPI_COMM_WORLD));

  if(solution_norm > 1.e-12)
  {
    error =  error_norm/solution_norm;
    relative_error = true;
  }
  else
  {
    error = error_norm;
    relative_error = false;
  }
}

template<int dim>
class ErrorCalculator
{
public:
  ErrorCalculator()
    :
    error_counter(0)
  {}

  void setup(DoFHandler<dim> const                                        &dof_handler_velocity_in,
             DoFHandler<dim> const                                        &dof_handler_pressure_in,
             Mapping<dim> const                                           &mapping_in,
             std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in,
             ErrorCalculationData const                                   &error_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    analytical_solution = analytical_solution_in;
    error_data = error_data_in;
  }

  void evaluate(parallel::distributed::Vector<double> const  &velocity,
                parallel::distributed::Vector<double> const  &pressure,
                double const                                 &time,
                int const                                    &time_step_number)
  {
    const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

    if(error_data.analytical_solution_available == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        if((time > (error_data.error_calc_start_time + error_counter*error_data.error_calc_interval_time - EPSILON)) )
        {
          ConditionalOStream pcout(std::cout,
            Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
          pcout << std::endl << "Calculate error at time t = "
                << std::scientific << std::setprecision(4) << time << ":" << std::endl;

          do_evaluate(velocity,pressure,time);

          ++error_counter;
        }
      }
      else // steady problem (time_step_number = -1)
      {
        ConditionalOStream pcout(std::cout,
            Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
        pcout << std::endl << "Calculate error for "
              << (error_counter == 0 ? "initial" : "solution") << " data"
              << std::endl;

        do_evaluate(velocity,pressure,time);

        ++error_counter;
      }
    }
  }

private:
  unsigned int error_counter;

  SmartPointer< DoFHandler<dim> const > dof_handler_velocity;
  SmartPointer< DoFHandler<dim> const > dof_handler_pressure;
  SmartPointer< Mapping<dim> const > mapping;

  std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  ErrorCalculationData error_data;

  void do_evaluate(parallel::distributed::Vector<double> const  &velocity,
                   parallel::distributed::Vector<double> const  &pressure,
                   double const                                 &time)
  {
    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    // velocity
    analytical_solution->velocity->set_time(time);
    double error_velocity = 1.0;
    bool relative_velocity = true;

    calculate_L2_error<dim>(*dof_handler_velocity,
                            *mapping,
                            velocity,
                            analytical_solution->velocity,
                            error_velocity,
                            relative_velocity);

    pcout << ((relative_velocity == true) ? "  Relative " : "  ABSOLUTE ") << "error (L2-norm) velocity u: "
          << std::scientific << std::setprecision(5) << error_velocity << std::endl;

    // pressure
    analytical_solution->pressure->set_time(time);
    double error_pressure = 1.0;
    bool relative_pressure = true;

    calculate_L2_error<dim>(*dof_handler_pressure,
                            *mapping,
                            pressure,
                            analytical_solution->pressure,
                            error_pressure,
                            relative_pressure);

    pcout << ((relative_pressure == true) ? "  Relative " : "  ABSOLUTE ") << "error (L2-norm) pressure p: "
          << std::scientific << std::setprecision(5) << error_pressure << std::endl;
  }
};



#endif /* INCLUDE_ERRORCALCULATIONNAVIERSTOKES_H_ */
