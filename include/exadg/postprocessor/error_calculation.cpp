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
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
template<int dim, typename VectorType>
double
calculate_error(MPI_Comm const &                             mpi_comm,
                bool const &                                 relative_error,
                dealii::DoFHandler<dim> const &              dof_handler,
                dealii::Mapping<dim> const &                 mapping,
                VectorType const &                           numerical_solution,
                std::shared_ptr<dealii::Function<dim>> const analytical_solution,
                double const &                               time,
                dealii::VectorTools::NormType const &        norm_type,
                unsigned int const                           additional_quadrature_points = 3)
{
  double error = 1.0;
  analytical_solution->set_time(time);

  dealii::LinearAlgebra::distributed::Vector<double> numerical_solution_double;
  numerical_solution_double = numerical_solution;
  numerical_solution_double.update_ghost_values();

  // calculate error norm
  dealii::Vector<double> error_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
  dealii::VectorTools::integrate_difference(mapping,
                                            dof_handler,
                                            numerical_solution_double,
                                            *analytical_solution,
                                            error_norm_per_cell,
                                            dealii::QGauss<dim>(dof_handler.get_fe().degree +
                                                                additional_quadrature_points),
                                            norm_type);

  double error_norm =
    std::sqrt(dealii::Utilities::MPI::sum(error_norm_per_cell.norm_sqr(), mpi_comm));

  if(relative_error == true)
  {
    // calculate solution norm
    dealii::Vector<double> solution_norm_per_cell(dof_handler.get_triangulation().n_active_cells());
    dealii::LinearAlgebra::distributed::Vector<double> zero_solution;
    zero_solution.reinit(numerical_solution);
    zero_solution.update_ghost_values();

    dealii::VectorTools::integrate_difference(mapping,
                                              dof_handler,
                                              zero_solution,
                                              *analytical_solution,
                                              solution_norm_per_cell,
                                              dealii::QGauss<dim>(dof_handler.get_fe().degree +
                                                                  additional_quadrature_points),
                                              norm_type);

    double solution_norm =
      std::sqrt(dealii::Utilities::MPI::sum(solution_norm_per_cell.norm_sqr(), mpi_comm));

    AssertThrow(solution_norm > 1.e-15,
                dealii::ExcMessage(
                  "Cannot compute relative error since norm of solution tends to zero."));

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
  : mpi_comm(comm), clear_files_L2(true), clear_files_H1_seminorm(true)
{
}

template<int dim, typename Number>
void
ErrorCalculator<dim, Number>::setup(dealii::DoFHandler<dim> const &   dof_handler_in,
                                    dealii::Mapping<dim> const &      mapping_in,
                                    ErrorCalculationData<dim> const & error_data_in)
{
  dof_handler = &dof_handler_in;
  mapping     = &mapping_in;
  error_data  = error_data_in;

  time_control.setup(error_data_in.time_control_data);

  if(error_data.analytical_solution && error_data.write_errors_to_file)
    create_directories(error_data.directory, mpi_comm);
}

template<int dim, typename Number>
void
ErrorCalculator<dim, Number>::evaluate(VectorType const & solution,
                                       double const       time,
                                       bool const         unsteady)
{
  AssertThrow(error_data.analytical_solution,
              dealii::ExcMessage("Function can only be called if analytical solution is given."));

  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  if(unsteady)
  {
    pcout << std::endl
          << "Calculate error for " << error_data.name << " at time t = " << std::scientific
          << std::setprecision(4) << time << ":" << std::endl;
  }
  else
  {
    pcout << std::endl
          << "Calculate error for " << error_data.name << " for "
          << (time_control.get_counter() == 0 ? "initial" : "solution") << " data:" << std::endl;
  }

  do_evaluate(solution, time);
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
                                            dealii::VectorTools::L2_norm);

  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);
  pcout << ((relative == true) ? "  Relative " : "  Absolute ")
        << "error (L2-norm): " << std::scientific << std::setprecision(5) << error << std::endl;

  if(error_data.write_errors_to_file)
  {
    // write output file
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::string filename_base =
        error_data.name + "_L2" + ((relative == true) ? "_relative" : "_absolute");

      std::string filename =
        filename_from_filename_base(error_data.directory, filename_base, clear_files_L2);

      std::ofstream f;
      if(clear_files_L2 == true)
      {
        f.open(filename.c_str(), std::ios::trunc);

        f << "  Time                Error" << std::endl;

        clear_files_L2 = false;
      }
      else
      {
        f.open(filename.c_str(), std::ios::app);
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
                                              dealii::VectorTools::H1_seminorm);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);
    pcout << ((relative == true) ? "  Relative " : "  Absolute ")
          << "error (H1-seminorm): " << std::scientific << std::setprecision(5) << error
          << std::endl;

    if(error_data.write_errors_to_file)
    {
      // write output file
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        std::string filename_base =
          error_data.name + "_H1_seminorm" + ((relative == true) ? "_relative" : "_absolute");

        std::string filename =
          filename_from_filename_base(error_data.directory, filename_base, clear_files_H1_seminorm);

        std::ofstream f;
        if(clear_files_H1_seminorm == true)
        {
          f.open(filename.c_str(), std::ios::trunc);

          f << "  Time                Error" << std::endl;

          clear_files_H1_seminorm = false;
        }
        else
        {
          f.open(filename.c_str(), std::ios::app);
        }

        unsigned int precision = 12;
        f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
          << std::setw(precision + 8) << error << std::endl;
      }
    }
  }
}

template<int dim, typename Number>
std::string
ErrorCalculator<dim, Number>::filename_from_filename_base(std::string const & directory,
                                                          std::string const & filename_base,
                                                          bool const          initial_call)
{
  // expected file beginning
  std::string filename = directory + "run_";

  // get all files matching the filename_base and
  // their run_id, then compare path ends
  std::filesystem::path const fs_directory{directory};
  std::vector<std::string>    matching_entries;
  std::vector<unsigned int>   run_ids;
  for(auto const & fs_dir_entry : std::filesystem::directory_iterator{fs_directory})
  {
    std::string dir_entry = fs_dir_entry.path().string();

    if(dir_entry.size() >= filename_base.size())
    {
      bool base_match = 0 == dir_entry.compare(dir_entry.size() - filename_base.size(),
                                               filename_base.size(),
                                               filename_base);
      if(base_match)
      {
        std::string run_id_string =
          dir_entry.substr(filename.size(), dir_entry.size() - filename.size());
        run_id_string = run_id_string.substr(0, run_id_string.find_first_of("_"));

        run_ids.push_back(dealii::Utilities::string_to_int(run_id_string));
        matching_entries.push_back(dir_entry);
      }
    }
  }

  unsigned int run_id;
  if(initial_call)
  {
    if(0 == run_ids.size())
    {
      // no existing files matching the filename_base
      run_id = 0;
    }
    else
    {
      // get new run_id
      run_id = 1 + *std::max_element(run_ids.begin(), run_ids.end());
    }
  }
  else
  {
    // append to the most recent file
    AssertThrow(run_ids.size() > 0,
                dealii::ExcMessage("Could not detect run_ids from files in " + directory));
    run_id = *std::max_element(run_ids.begin(), run_ids.end());
  }

  return filename + std::to_string(run_id) + "_" + filename_base;
}

template class ErrorCalculator<2, float>;
template class ErrorCalculator<2, double>;

template class ErrorCalculator<3, float>;
template class ErrorCalculator<3, double>;

} // namespace ExaDG
