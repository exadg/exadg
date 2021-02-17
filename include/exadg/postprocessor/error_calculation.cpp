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

// C/C++
#include <fstream>

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/postprocessor/error_calculation.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename VectorType>
double
calculate_error(MPI_Comm const &                     mpi_comm,
                bool const &                         relative_error,
                DoFHandler<dim> const &              dof_handler,
                Mapping<dim> const &                 mapping,
                VectorType const &                   numerical_solution,
                std::shared_ptr<Function<dim>> const analytical_solution,
                double const &                       time,
                VectorTools::NormType const &        norm_type,
                unsigned int const                   additional_quadrature_points = 3)
{
  double error = 1.0;
  analytical_solution->set_time(time);

  LinearAlgebra::distributed::Vector<double> numerical_solution_double;
  numerical_solution_double = numerical_solution;
  numerical_solution_double.update_ghost_values();

  // calculate error norm
  Vector<double> error_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    numerical_solution_double,
                                    *analytical_solution,
                                    error_norm_per_cell,
                                    QGauss<dim>(dof_handler.get_fe().degree +
                                                additional_quadrature_points),
                                    norm_type);

  double error_norm = std::sqrt(Utilities::MPI::sum(error_norm_per_cell.norm_sqr(), mpi_comm));

  if(relative_error == true)
  {
    // calculate solution norm
    Vector<double> solution_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
    LinearAlgebra::distributed::Vector<double> zero_solution;
    zero_solution.reinit(numerical_solution);
    zero_solution.update_ghost_values();

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      zero_solution,
                                      *analytical_solution,
                                      solution_norm_per_cell,
                                      QGauss<dim>(dof_handler.get_fe().degree +
                                                  additional_quadrature_points),
                                      norm_type);

    double solution_norm =
      std::sqrt(Utilities::MPI::sum(solution_norm_per_cell.norm_sqr(), mpi_comm));

    AssertThrow(solution_norm > 1.e-15,
                ExcMessage("Cannot compute relative error since norm of solution tends to zero."));

    error = error_norm / solution_norm;
  }
  else // absolute error
  {
    error = error_norm;
  }

  return error;
}

template<int dim, typename Number>
ErrorCalculator<dim, Number>::ErrorCalculator(MPI_Comm const & comm)
  : mpi_comm(comm),
    error_counter(0),
    reset_counter(true),
    clear_files_L2(true),
    clear_files_H1_seminorm(true)
{
}

template<int dim, typename Number>
void
ErrorCalculator<dim, Number>::setup(DoFHandler<dim> const &           dof_handler_in,
                                    Mapping<dim> const &              mapping_in,
                                    ErrorCalculationData<dim> const & error_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  error_data  = error_data_in;
}

template<int dim, typename Number>
void
ErrorCalculator<dim, Number>::evaluate(VectorType const & solution,
                                       double const &     time,
                                       int const &        time_step_number)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  if(error_data.analytical_solution_available == true)
  {
    if(time_step_number >= 0) // unsteady problem
    {
      // small number which is much smaller than the time step size
      const double EPSILON = 1.0e-10;

      // In the first time step, the current time might be larger than output_start_time. In that
      // case, we first have to reset the counter in order to avoid that output is written every
      // time step.
      if(reset_counter)
      {
        error_counter += int((time - error_data.error_calc_start_time + EPSILON) /
                             error_data.error_calc_interval_time);
        reset_counter = false;
      }

      if((time > (error_data.error_calc_start_time +
                  error_counter * error_data.error_calc_interval_time - EPSILON)))
      {
        pcout << std::endl
              << "Calculate error for " << error_data.name << " at time t = " << std::scientific
              << std::setprecision(4) << time << ":" << std::endl;

        do_evaluate(solution, time);

        ++error_counter;
      }
    }
    else // steady problem (time_step_number = -1)
    {
      pcout << std::endl
            << "Calculate error for " << error_data.name << " for "
            << (error_counter == 0 ? "initial" : "solution") << " data:" << std::endl;

      do_evaluate(solution, time);

      ++error_counter;
    }
  }
}

template<int dim, typename Number>
void
ErrorCalculator<dim, Number>::do_evaluate(VectorType const & solution_vector, double const time)
{
  bool relative = error_data.calculate_relative_errors;

  double const error = calculate_error<dim>(mpi_comm,
                                            relative,
                                            *dof_handler,
                                            *mapping,
                                            solution_vector,
                                            error_data.analytical_solution,
                                            time,
                                            VectorTools::L2_norm);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);
  pcout << ((relative == true) ? "  Relative " : "  Absolute ")
        << "error (L2-norm): " << std::scientific << std::setprecision(5) << error << std::endl;

  if(error_data.write_errors_to_file)
  {
    // write output file
    if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::ostringstream filename;
      filename << error_data.folder + error_data.name + "_L2";

      std::ofstream f;
      if(clear_files_L2 == true)
      {
        f.open(filename.str().c_str(), std::ios::trunc);

        f << "  Time                Error" << std::endl;

        clear_files_L2 = false;
      }
      else
      {
        f.open(filename.str().c_str(), std::ios::app);
      }

      unsigned int precision = 12;
      f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
        << std::setw(precision + 8) << error << std::endl;
    }
  }

  // H1-seminorm
  if(error_data.calculate_H1_seminorm_error)
  {
    double const error = calculate_error<dim>(mpi_comm,
                                              relative,
                                              *dof_handler,
                                              *mapping,
                                              solution_vector,
                                              error_data.analytical_solution,
                                              time,
                                              VectorTools::H1_seminorm);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);
    pcout << ((relative == true) ? "  Relative " : "  Absolute ")
          << "error (H1-seminorm): " << std::scientific << std::setprecision(5) << error
          << std::endl;

    if(error_data.write_errors_to_file)
    {
      // write output file
      if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        std::ostringstream filename;
        filename << error_data.folder + error_data.name + "_H1_seminorm";

        std::ofstream f;
        if(clear_files_H1_seminorm == true)
        {
          f.open(filename.str().c_str(), std::ios::trunc);

          f << "  Time                Error" << std::endl;

          clear_files_H1_seminorm = false;
        }
        else
        {
          f.open(filename.str().c_str(), std::ios::app);
        }

        unsigned int precision = 12;
        f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
          << std::setw(precision + 8) << error << std::endl;
      }
    }
  }
}

template class ErrorCalculator<2, float>;
template class ErrorCalculator<2, double>;

template class ErrorCalculator<3, float>;
template class ErrorCalculator<3, double>;

} // namespace ExaDG
