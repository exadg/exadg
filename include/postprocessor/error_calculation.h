/*
 * error_calculation.h
 *
 *  Created on: Feb 20, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_H_

#include "postprocessor/calculate_l2_error.h"
#include "postprocessor/error_calculation_data.h"

template<int dim, typename Number>
class ErrorCalculator
{
public:
  ErrorCalculator()
    :
    error_counter(0)
  {}

  void setup(DoFHandler<dim> const           &dof_handler_in,
             Mapping<dim> const              &mapping_in,
             std::shared_ptr<Function<dim> > analytical_solution_in,
             ErrorCalculationData const      &error_data_in)
  {
    dof_handler = &dof_handler_in;
    mapping = &mapping_in;
    analytical_solution = analytical_solution_in;
    error_data = error_data_in;
  }

  void evaluate(parallel::distributed::Vector<Number> const  &solution,
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

          do_evaluate(solution,time);

          ++error_counter;
        }
      }
      else // steady problem (time_step_number = -1)
      {
        pcout << std::endl << "Calculate error for "
              << (error_counter == 0 ? "initial" : "solution") << " data"
              << std::endl;

        do_evaluate(solution,time);

        ++error_counter;
      }
    } 
    else // no analytical solution available: simply print L2-norm
    {
      auto analytical_solution = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(1));
        
      bool relative = false;
      double const error = calculate_L2_error<dim>(relative,
                                                   *dof_handler,
                                                   *mapping,
                                                   solution,
                                                   analytical_solution,
                                                   0.0);
  
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << "  L2-norm: "
            << std::scientific << std::setprecision(15) << error << std::endl;
    }
  }

private:
  unsigned int error_counter;

  SmartPointer< DoFHandler<dim> const > dof_handler;
  SmartPointer< Mapping<dim> const > mapping;

  std::shared_ptr<Function<dim> > analytical_solution;

  ErrorCalculationData error_data;

  void do_evaluate(parallel::distributed::Vector<double> const &solution_vector,
                   double const                                time) const
  {
    bool relative = true;
    double const error = calculate_L2_error<dim>(relative,
                                                 *dof_handler,
                                                 *mapping,
                                                 solution_vector,
                                                 analytical_solution,
                                                 time);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << ((relative == true) ? "  Relative " : "  ABSOLUTE ") << "error (L2-norm): "
          << std::scientific << std::setprecision(5) << error << std::endl;
  }
};

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_H_ */
