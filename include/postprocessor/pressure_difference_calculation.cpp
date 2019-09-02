/*
 * pressure_difference_calculation.cpp
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

#include "pressure_difference_calculation.h"

#include "postprocessor/evaluate_solution_in_given_point.h"

template<int dim, typename Number>
PressureDifferenceCalculator<dim, Number>::PressureDifferenceCalculator()
  : clear_files_pressure_difference(true)
{
}

template<int dim, typename Number>
void
PressureDifferenceCalculator<dim, Number>::setup(
  DoFHandler<dim> const &             dof_handler_pressure_in,
  Mapping<dim> const &                mapping_in,
  PressureDifferenceData<dim> const & pressure_difference_data_in)
{
  dof_handler_pressure     = &dof_handler_pressure_in;
  mapping                  = &mapping_in;
  pressure_difference_data = pressure_difference_data_in;
}

template<int dim, typename Number>
void
PressureDifferenceCalculator<dim, Number>::evaluate(VectorType const & pressure,
                                                    double const &     time) const
{
  if(pressure_difference_data.calculate_pressure_difference == true)
  {
    Number pressure_1 = 0.0, pressure_2 = 0.0;

    Point<dim> point_1, point_2;
    point_1 = pressure_difference_data.point_1;
    point_2 = pressure_difference_data.point_2;

    evaluate_scalar_quantity_in_point(
      *dof_handler_pressure, *mapping, pressure, point_1, pressure_1);
    evaluate_scalar_quantity_in_point(
      *dof_handler_pressure, *mapping, pressure, point_2, pressure_2);

    Number const pressure_difference = pressure_1 - pressure_2;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::string filename = pressure_difference_data.filename;

      unsigned int precision = 12;

      std::ofstream f;
      if(clear_files_pressure_difference)
      {
        f.open(filename.c_str(), std::ios::trunc);

        // clang-format off
        f << std::setw(precision + 8) << std::left << "time t"
          << std::setw(precision + 8) << std::left << "pressure difference"
          << std::endl;
        // clang-format on

        clear_files_pressure_difference = false;
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
}

template class PressureDifferenceCalculator<2, float>;
template class PressureDifferenceCalculator<2, double>;

template class PressureDifferenceCalculator<3, float>;
template class PressureDifferenceCalculator<3, double>;
