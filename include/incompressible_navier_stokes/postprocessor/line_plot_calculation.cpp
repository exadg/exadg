/*
 * line_plot_calculation.cpp
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#include "line_plot_calculation.h"

template<int dim, typename Number>
LinePlotCalculator<dim, Number>::LinePlotCalculator() : clear_files(true)
{
}

template<int dim, typename Number>
void
LinePlotCalculator<dim, Number>::setup(DoFHandler<dim> const &   dof_handler_velocity_in,
                                       DoFHandler<dim> const &   dof_handler_pressure_in,
                                       Mapping<dim> const &      mapping_in,
                                       LinePlotData<dim> const & line_plot_data_in)
{
  dof_handler_velocity = &dof_handler_velocity_in;
  dof_handler_pressure = &dof_handler_pressure_in;
  mapping              = &mapping_in;
  data                 = line_plot_data_in;
}

template<int dim, typename Number>
void
LinePlotCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                          VectorType const & pressure) const
{
  if(data.write_output == true)
  {
    // precision
    unsigned int const precision = data.precision;

    // loop over all lines
    for(typename std::vector<Line<dim>>::const_iterator line = data.lines.begin();
        line != data.lines.end();
        ++line)
    {
      // store all points along current line in a vector
      unsigned int            n_points = line->n_points;
      std::vector<Point<dim>> points(n_points);

      // we consider straight lines with an equidistant distribution of points along the line
      for(unsigned int i = 0; i < n_points; ++i)
        points[i] = line->begin + double(i) / double(n_points - 1) * (line->end - line->begin);

      // filename prefix for current line
      std::string filename_prefix = data.filename_prefix + "_" + line->name;

      // write output for all specified quantities
      for(std::vector<Quantity *>::const_iterator quantity = line->quantities.begin();
          quantity != line->quantities.end();
          ++quantity)
      {
        if((*quantity)->type == QuantityType::Velocity)
        {
          std::vector<Tensor<1, dim, Number>> solution_vector(n_points);

          // calculate velocity for all points along line
          for(unsigned int i = 0; i < n_points; ++i)
          {
            Tensor<1, dim, Number> u;
            evaluate_vectorial_quantity_in_point(
              *dof_handler_velocity, *mapping, velocity, points[i], u);
            solution_vector[i] = u;
          }

          // write output to file
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::string filename = filename_prefix + "_velocity" + ".txt";

            std::ofstream f;
            if(clear_files)
            {
              f.open(filename.c_str(), std::ios::trunc);
              //                clear_files = false; // TODO
            }
            else
            {
              f.open(filename.c_str(), std::ios::app);
            }

            // headline
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << "x_" + Utilities::int_to_string(d + 1);
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << "u_" + Utilities::int_to_string(d + 1);
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
              *dof_handler_pressure, *mapping, pressure, points[i], p);
            solution_vector[i] = p;
          }

          // write output to file
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::string filename = filename_prefix + "_pressure" + ".txt";

            std::ofstream f;
            if(clear_files)
            {
              f.open(filename.c_str(), std::ios::trunc);
              // TODO: overwrite the same files
              //                clear_files = false;
            }
            else
            {
              f.open(filename.c_str(), std::ios::app);
            }

            // headline
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << "x_" + Utilities::int_to_string(d + 1);
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
  }     // write_output == true
}

template class LinePlotCalculator<2, float>;
template class LinePlotCalculator<2, double>;

template class LinePlotCalculator<3, float>;
template class LinePlotCalculator<3, double>;
