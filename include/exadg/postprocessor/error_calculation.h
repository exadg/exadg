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

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
template<int dim>
struct ErrorCalculationData
{
  ErrorCalculationData()
    : calculate_relative_errors(true),
      calculate_H1_seminorm_error(false),
      write_errors_to_file(false),
      spatially_weight_error(false),
      weight(nullptr),
      directory("output/"),
      name("all fields")
  {
  }

  void
  print(dealii::ConditionalOStream & pcout, bool unsteady)
  {
    print_parameter(pcout, "Error calculation", unsteady == true and analytical_solution);
    if(unsteady == true and time_control_data.is_active)
    {
      print(pcout, unsteady, time_control_data);
      print_parameter(pcout, "Calculate relative errors", calculate_relative_errors);
      print_parameter(pcout, "Calculate H1-seminorm error", calculate_H1_seminorm_error);
      print_parameter(pcout, "Write errors to file", write_errors_to_file);
      if(write_errors_to_file)
        print_parameter(pcout, "Directory", directory);
      print_parameter(pcout, "Name", name);
    }
  }

  std::shared_ptr<dealii::Function<dim>> analytical_solution;

  // relative or absolute errors?
  // If calculate_relative_errors == false, this implies that absolute errors are calculated
  bool calculate_relative_errors;

  // by default, only the L2-error is computed. Other norms have to be explicitly specified by the
  // user.
  bool calculate_H1_seminorm_error;

  // data used to control the output
  TimeControlData time_control_data;

  // write errors to file?
  bool write_errors_to_file;

  // If true, a spatially weighted norm is computed.
  bool spatially_weight_error;
  // Weight used to compute spatially weighted error.
  std::shared_ptr<dealii::Function<dim>> weight;

  // directory and name (used as filename and as identifier for screen output)
  std::string directory;
  std::string name;
};

template<int dim, typename Number>
class ErrorCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  ErrorCalculator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const &   dof_handler,
        dealii::Mapping<dim> const &      mapping,
        ErrorCalculationData<dim> const & error_data);

  void
  evaluate(VectorType const & solution, double const time, bool const unsteady);

  TimeControl time_control;

private:
  void
  do_evaluate(VectorType const & solution_vector, double const time);

  MPI_Comm const mpi_comm;

  bool clear_files_L2, clear_files_H1_seminorm;

  dealii::SmartPointer<dealii::DoFHandler<dim> const> dof_handler;
  dealii::SmartPointer<dealii::Mapping<dim> const>    mapping;

  ErrorCalculationData<dim> error_data;
};

} // namespace ExaDG

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ERROR_CALCULATION_H_ */
