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

namespace IncNS
{
template<int dim, typename Number>
class ErrorCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  ErrorCalculator()
    : clear_files_velocity(true),
      clear_files_pressure(true),
      clear_files_velocity_H1_seminorm(true),
      counter(0)
  {
  }

  void
  setup(DoFHandler<dim> const &                  dof_handler_velocity_in,
        DoFHandler<dim> const &                  dof_handler_pressure_in,
        Mapping<dim> const &                     mapping_in,
        std::shared_ptr<AnalyticalSolution<dim>> analytical_solution_in,
        ErrorCalculationData const &             error_data_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping              = &mapping_in;
    analytical_solution  = analytical_solution_in;
    error_data           = error_data_in;
  }

public:
  void
  evaluate(VectorType const & velocity,
           VectorType const & pressure,
           double const &     time,
           int const &        time_step_number)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    if(error_data.analytical_solution_available == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        if(error_has_to_be_calculated(time, time_step_number))
        {
          pcout << std::endl
                << "Calculate error at time t = " << std::scientific << std::setprecision(4) << time
                << ":" << std::endl;

          do_evaluate(velocity, pressure, time);
        }
      }
      else // steady problem (time_step_number = -1)
      {
        pcout << std::endl
              << "Calculate error for " << (counter == 0 ? "initial" : "solution") << " data"
              << std::endl;

        do_evaluate(velocity, pressure, time);

        ++counter;
      }
    }
  }

private:
  bool         clear_files_velocity, clear_files_pressure;
  bool         clear_files_velocity_H1_seminorm;
  unsigned int counter;

  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const>    mapping;

  std::shared_ptr<AnalyticalSolution<dim>> analytical_solution;

  ErrorCalculationData error_data;

  bool
  error_has_to_be_calculated(double const & time, int const & time_step_number)
  {
    double const EPSILON = 1.0e-10; // small number which is much smaller than the time step size
    if((time > (error_data.error_calc_start_time + counter * error_data.error_calc_interval_time -
                EPSILON)))
    {
      counter++;
      return true;
    }
    else if(time > error_data.error_calc_start_time &&
            time_step_number % error_data.calculate_every_time_steps == 0)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  void
  do_evaluate(VectorType const & velocity, VectorType const & pressure, double const & time)
  {
    bool const relative_errors = error_data.calculate_relative_errors;

    // velocity
    LinearAlgebra::distributed::Vector<double> velocity_double;
    velocity_double = velocity;

    // L2-norm
    double const error_velocity = calculate_error<dim>(relative_errors,
                                                       *dof_handler_velocity,
                                                       *mapping,
                                                       velocity_double,
                                                       analytical_solution->velocity,
                                                       time,
                                                       VectorTools::L2_norm);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << ((relative_errors == true) ? "  Relative " : "  Absolute ")
          << "L2-error velocity u: " << std::scientific << std::setprecision(5) << error_velocity
          << std::endl;

    if(error_data.write_errors_to_file)
    {
      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ostringstream filename;
        filename << error_data.filename_prefix + "_velocity_L2";

        std::ofstream f;
        if(clear_files_velocity == true)
        {
          f.open(filename.str().c_str(), std::ios::trunc);
          if(relative_errors == true)
            f << "Relative L2-error velocity || u - u_h ||_Omega / || u_h ||_Omega:" << std::endl;
          else
            f << "Absolute L2-error velocity || u - u_h ||_Omega:" << std::endl;

          f << std::endl << "  Time                Error" << std::endl;

          clear_files_velocity = false;
        }
        else
        {
          f.open(filename.str().c_str(), std::ios::app);
        }

        unsigned int precision = 12;
        f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
          << std::setw(precision + 8) << error_velocity << std::endl;
      }
    }

    // H1-seminorm
    if(error_data.calculate_H1_seminorm_velocity)
    {
      double const error_velocity = calculate_error<dim>(relative_errors,
                                                         *dof_handler_velocity,
                                                         *mapping,
                                                         velocity_double,
                                                         analytical_solution->velocity,
                                                         time,
                                                         VectorTools::H1_seminorm);

      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << ((relative_errors == true) ? "  Relative " : "  Absolute ")
            << "H1-seminorm error velocity u: " << std::scientific << std::setprecision(5)
            << error_velocity << std::endl;

      if(error_data.write_errors_to_file)
      {
        // write output file
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ostringstream filename;
          filename << error_data.filename_prefix + "_velocity_H1_seminorm";

          std::ofstream f;
          if(clear_files_velocity_H1_seminorm == true)
          {
            f.open(filename.str().c_str(), std::ios::trunc);
            if(relative_errors == true)
              f << "Relative H1-seminorm error velocity || (grad(u) - (grad(u_h) ||_Omega / || grad(u_h) ||_Omega:"
                << std::endl;
            else
              f << "Absolute H1-seminorm error velocity || grad(u) - grad(u_h) ||_Omega:"
                << std::endl;

            f << std::endl << "  Time                Error" << std::endl;

            clear_files_velocity_H1_seminorm = false;
          }
          else
          {
            f.open(filename.str().c_str(), std::ios::app);
          }

          unsigned int precision = 12;
          f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
            << std::setw(precision + 8) << error_velocity << std::endl;
        }
      }
    }


    // pressure
    LinearAlgebra::distributed::Vector<double> pressure_double;
    pressure_double = pressure;

    // L2-norm
    double const error_pressure = calculate_error<dim>(relative_errors,
                                                       *dof_handler_pressure,
                                                       *mapping,
                                                       pressure_double,
                                                       analytical_solution->pressure,
                                                       time,
                                                       VectorTools::L2_norm);

    pcout << ((relative_errors == true) ? "  Relative " : "  Absolute ")
          << "L2-error pressure p: " << std::scientific << std::setprecision(5) << error_pressure
          << std::endl;

    if(error_data.write_errors_to_file)
    {
      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ostringstream filename;
        filename << error_data.filename_prefix + "_pressure_L2";

        std::ofstream f;
        if(clear_files_pressure == true)
        {
          f.open(filename.str().c_str(), std::ios::trunc);
          if(relative_errors == true)
            f << "Relative L2-error pressure || p - p_h ||_Omega / || p_h ||_Omega:" << std::endl;
          else
            f << "Absolute L2-error pressure || p - p_h ||_Omega:" << std::endl;

          f << std::endl << "  Time                Error" << std::endl;

          clear_files_pressure = false;
        }
        else
        {
          f.open(filename.str().c_str(), std::ios::app);
        }

        unsigned int precision = 12;
        f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
          << std::setw(precision + 8) << error_pressure << std::endl;
      }
    }
  }
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_NAVIER_STOKES_H_ */
