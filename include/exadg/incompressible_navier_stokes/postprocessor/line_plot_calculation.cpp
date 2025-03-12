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
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation.h>
#include <exadg/postprocessor/solution_interpolation.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
LinePlotCalculator<dim, Number>::LinePlotCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true)
{
}

template<int dim, typename Number>
void
LinePlotCalculator<dim, Number>::setup(dealii::DoFHandler<dim> const & dof_handler_velocity_in,
                                       dealii::DoFHandler<dim> const & dof_handler_pressure_in,
                                       dealii::Mapping<dim> const &    mapping_in,
                                       LinePlotData<dim> const &       line_plot_data_in)
{
  dof_handler_velocity = &dof_handler_velocity_in;
  dof_handler_pressure = &dof_handler_pressure_in;
  mapping              = &mapping_in;
  data                 = line_plot_data_in;

  time_control.setup(line_plot_data_in.time_control_data);

  if(line_plot_data_in.time_control_data.is_active)
    create_directories(line_plot_data_in.directory, mpi_comm);
}

template<int dim, typename Number>
void
LinePlotCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                          VectorType const & pressure) const
{
  // precision
  unsigned int const precision = data.precision;

  // loop over all lines
  for(typename std::vector<std::shared_ptr<Line<dim>>>::const_iterator line = data.lines.begin();
      line != data.lines.end();
      ++line)
  {
    // store all points along current line in a vector
    unsigned int                    n_points = (*line)->n_points;
    std::vector<dealii::Point<dim>> points(n_points);

    // we consider straight lines with an equidistant distribution of points along the line
    for(unsigned int i = 0; i < n_points; ++i)
      points[i] =
        (*line)->begin + double(i) / double(n_points - 1) * ((*line)->end - (*line)->begin);

    // filename prefix for current line
    std::string filename_prefix = data.directory + (*line)->name;

    // write output for all specified quantities
    for(std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
          (*line)->quantities.begin();
        quantity != (*line)->quantities.end();
        ++quantity)
    {
      if((*quantity)->type == QuantityType::Velocity)
      {
        std::vector<dealii::Tensor<1, dim, Number>> solution_vector(n_points);

        // calculate velocity for all points along line
        for(unsigned int i = 0; i < n_points; ++i)
        {
          dealii::Tensor<1, dim, Number> u;
          evaluate_vectorial_quantity_in_point(
            u, *dof_handler_velocity, *mapping, velocity, points[i], mpi_comm);
          solution_vector[i] = u;
        }

        // write output to file
        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        {
          std::string filename = filename_prefix + "_velocity" + ".txt";

          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          // headline
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "u_" + dealii::Utilities::int_to_string(d + 1);
          f << std::endl;

          // loop over all points
          for(unsigned int i = 0; i < n_points; ++i)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << points[i][d];
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << solution_vector[i][d];
            f << std::endl;
          }
          f.close();
        }
      }
      else if((*quantity)->type == QuantityType::Pressure)
      {
        std::vector<Number> solution_vector(n_points);

        // calculate pressure for all points along line
        for(unsigned int i = 0; i < n_points; ++i)
        {
          Number p;
          evaluate_scalar_quantity_in_point(
            p, *dof_handler_pressure, *mapping, pressure, points[i], mpi_comm);
          solution_vector[i] = p;
        }

        // write output to file
        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        {
          std::string filename = filename_prefix + "_pressure" + ".txt";

          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          // headline
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);
          f << std::setw(precision + 8) << std::left << "p";
          f << std::endl;

          // loop over all points
          for(unsigned int i = 0; i < n_points; ++i)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << points[i][d];
            f << std::setw(precision + 8) << std::left << solution_vector[i];
            f << std::endl;
          }
          f.close();
        }
      }
    } // loop over quantities
  }   // loop over lines
}

template class LinePlotCalculator<2, float>;
template class LinePlotCalculator<2, double>;

template class LinePlotCalculator<3, float>;
template class LinePlotCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
