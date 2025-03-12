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

// ExaDG
#include <exadg/postprocessor/pressure_difference_calculation.h>
#include <exadg/postprocessor/solution_interpolation.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
template<int dim>
void
PressureDifferenceData<dim>::print(dealii::ConditionalOStream & pcout, bool const unsteady) const
{
  if(time_control_data.is_active)
  {
    pcout << std::endl << "Pressure difference calculation" << std::endl;
    time_control_data.print(pcout, unsteady);

    print_parameter(pcout, "Point 1", point_1);
    print_parameter(pcout, "Point 2", point_2);

    print_parameter(pcout, "Directory", directory);
    print_parameter(pcout, "Filename", filename);
  }
}

template<int dim, typename Number>
PressureDifferenceCalculator<dim, Number>::PressureDifferenceCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true)
{
}

template<int dim, typename Number>
void
PressureDifferenceCalculator<dim, Number>::setup(
  dealii::DoFHandler<dim> const &     dof_handler_pressure_in,
  dealii::Mapping<dim> const &        mapping_in,
  PressureDifferenceData<dim> const & data_in)
{
  dof_handler_pressure = &dof_handler_pressure_in;
  mapping              = &mapping_in;
  data                 = data_in;

  time_control.setup(data.time_control_data);

  if(data.time_control_data.is_active)
    create_directories(data.directory, mpi_comm);
}

template<int dim, typename Number>
void
PressureDifferenceCalculator<dim, Number>::evaluate(VectorType const & pressure,
                                                    double const       time) const
{
  Number pressure_1 = 0.0, pressure_2 = 0.0;

  dealii::Point<dim> point_1, point_2;
  point_1 = data.point_1;
  point_2 = data.point_2;

  evaluate_scalar_quantity_in_point(
    pressure_1, *dof_handler_pressure, *mapping, pressure, point_1, mpi_comm);
  evaluate_scalar_quantity_in_point(
    pressure_2, *dof_handler_pressure, *mapping, pressure, point_2, mpi_comm);

  Number const pressure_difference = pressure_1 - pressure_2;

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::string filename = data.directory + data.filename;

    unsigned int precision = 12;

    std::ofstream f;
    if(clear_files)
    {
      f.open(filename.c_str(), std::ios::trunc);

      // clang-format off
        f << std::setw(precision + 8) << std::left << "time t"
          << std::setw(precision + 8) << std::left << "pressure difference"
          << std::endl;
      // clang-format on

      clear_files = false;
    }
    else
    {
      f.open(filename.c_str(), std::ios::app);
    }

    // clang-format off
      f << std::scientific << std::setprecision(precision)
        << std::setw(precision + 8) << std::left << time
        << std::setw(precision + 8) << std::left << pressure_difference
        << std::endl;
    // clang-format on

    f.close();
  }
}

template class PressureDifferenceCalculator<2, float>;
template class PressureDifferenceCalculator<2, double>;

template class PressureDifferenceCalculator<3, float>;
template class PressureDifferenceCalculator<3, double>;

} // namespace ExaDG
