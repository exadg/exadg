/*
 * ErrorCalculationNavierStokes.h
 *
 *  Created on: Oct 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_NAVIER_STOKES_H_

#include "postprocessor/calculate_l2_error.h"
#include "postprocessor/error_calculation_data.h"

#include "incompressible_navier_stokes/user_interface/analytical_solution.h"

template<int dim, typename Number>
class ErrorCalculator
{
public:
  ErrorCalculator()
    :
    error_counter(0)
  {}

  void setup(DoFHandler<dim> const                                 &dof_handler_velocity_in,
             DoFHandler<dim> const                                 &dof_handler_pressure_in,
             Mapping<dim> const                                    &mapping_in,
             std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution_in,
             ErrorCalculationData const                            &error_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    analytical_solution = analytical_solution_in;
    error_data = error_data_in;
  }

  void evaluate(parallel::distributed::Vector<Number> const  &velocity,
                parallel::distributed::Vector<Number> const  &pressure,
                double const                                 &time,
                int const                                    &time_step_number)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    if(error_data.analytical_solution_available == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
        if((time > (error_data.error_calc_start_time + error_counter*error_data.error_calc_interval_time - EPSILON)) )
        {
          pcout << std::endl << "Calculate error at time t = "
                << std::scientific << std::setprecision(4) << time << ":" << std::endl;

          do_evaluate(velocity,pressure,time);

          ++error_counter;
        }
      }
      else // steady problem (time_step_number = -1)
      {
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

  std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  ErrorCalculationData error_data;

  void do_evaluate(parallel::distributed::Vector<Number> const  &velocity,
                   parallel::distributed::Vector<Number> const  &pressure,
                   double const                                 &time)
  {
    // velocity
    bool relative_velocity = true;

    parallel::distributed::Vector<double> velocity_double;
    velocity_double = velocity;

    double const error_velocity = calculate_L2_error<dim>(relative_velocity,
                                                          *dof_handler_velocity,
                                                          *mapping,
                                                          velocity_double,
                                                          analytical_solution->velocity,
                                                          time);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << ((relative_velocity == true) ? "  Relative " : "  ABSOLUTE ") << "error (L2-norm) velocity u: "
          << std::scientific << std::setprecision(5) << error_velocity << std::endl;

    // pressure
    bool relative_pressure = true;

    parallel::distributed::Vector<double> pressure_double;
    pressure_double = pressure;

    double const error_pressure = calculate_L2_error<dim>(relative_pressure,
                                                          *dof_handler_pressure,
                                                          *mapping,
                                                          pressure_double,
                                                          analytical_solution->pressure,
                                                          time);

    pcout << ((relative_pressure == true) ? "  Relative " : "  ABSOLUTE ") << "error (L2-norm) pressure p: "
          << std::scientific << std::setprecision(5) << error_pressure << std::endl;
  }
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_NAVIER_STOKES_H_ */
